"""
Choropleth map from Eurostat / EU livestock GeoPackage C21.gpkg (or similar).

Default: first layer, value column `{layer}Density` when present (e.g. BovineDensity).
By default keeps only features whose centroid falls in a Europe / near-Atlantic bbox
(WGS84), so overseas territories do not shrink the map. Plot CRS defaults to EPSG:3035.

Linear maps use the 99th percentile as vmax (values above saturate at the top colour) so a
few extreme grid cells do not wash out the rest of Europe; use --full-range for raw min/max.

Run from repo root:
  python Agriculture/Auxiliaries/map_c21_gpkg.py
  python Agriculture/Auxiliaries/map_c21_gpkg.py --layer Pigs --field PigsDensity
  python Agriculture/Auxiliaries/map_c21_gpkg.py --max-rows 8000
  python Agriculture/Auxiliaries/map_c21_gpkg.py --no-bbox-filter --crs 4326
  python Agriculture/Auxiliaries/map_c21_gpkg.py --batch-counts

Requires: geopandas, matplotlib

`--batch-counts` writes five maps (head counts, not density): Bovine, Goats, Pigs (column A3100
in this GeoPackage), Poultry, Sheep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _list_layers(path: Path) -> list[str]:
    try:
        import geopandas as gpd

        df = gpd.list_layers(path)
        if df is not None and len(df) and "name" in df.columns:
            return df["name"].astype(str).tolist()
    except Exception:
        pass
    import sqlite3

    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features' ORDER BY table_name"
        ).fetchall()
        if rows:
            return [r[0] for r in rows]
    finally:
        con.close()
    try:
        import fiona

        return list(fiona.listlayers(path))
    except Exception:
        return []


def _resolve_layer(path: Path, layer: str | int | None) -> str:
    layers = _list_layers(path)
    if not layers:
        raise ValueError("No feature layers found in GeoPackage.")
    if layer is None:
        return layers[0]
    if isinstance(layer, int):
        return layers[layer]
    return str(layer)


# WGS84 (lon/lat): Europe + Azores / Canaries; excludes most overseas departments (e.g. GF, RE).
_DEFAULT_EUROPE_BBOX: tuple[float, float, float, float] = (-32.0, 26.0, 45.0, 72.0)

# Figure export: larger canvas + 300 dpi gives sharper PNGs (override with --dpi / --figsize).
_DEFAULT_FIGSIZE_IN: tuple[float, float] = (14.0, 11.0)
_DEFAULT_SAVE_DPI = 300

# GPKG layer name -> numeric column for livestock head counts (Pigs layer uses Eurostat A3100).
LIVESTOCK_COUNT_LAYERS: tuple[tuple[str, str], ...] = (
    ("Bovine", "Bovine"),
    ("Goats", "Goats"),
    ("Pigs", "A3100"),
    ("Poultry", "Poultry"),
    ("Sheep", "Sheep"),
)


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be west,south,east,north (four comma-separated numbers)")
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def _filter_europe_extent(gdf, west: float, south: float, east: float, north: float):
    """Keep rows whose geometry centroid (projected first) lies inside the WGS84 bbox."""
    c = gdf.to_crs(3035).geometry.centroid.to_crs(4326)
    mask = (c.x >= west) & (c.x <= east) & (c.y >= south) & (c.y <= north)
    return gdf.loc[mask].copy()


def _value_label(layer_name: str, value_col: str) -> str:
    """Legend/title: Pigs layer uses Eurostat column A3100 in this GPKG."""
    if layer_name == "Pigs" and value_col == "A3100":
        return "Pigs (A3100)"
    return value_col


def _infer_value_column(gdf, layer_name: str) -> str | None:
    cand = f"{layer_name}Density"
    if cand in gdf.columns:
        return cand
    for c in gdf.columns:
        if c == "geometry":
            continue
        if "Density" in c and gdf[c].dtype.kind in "iuf":
            return c
    skip = {"geometry", "ID", "id", "res"}
    for c in gdf.columns:
        if c in skip:
            continue
        if gdf[c].dtype.kind in "iuf":
            return c
    return None


def _plot(
    path: Path,
    layer: str | int | None,
    field: str | None,
    out: Path,
    max_rows: int | None,
    dpi: int,
    cmap: str,
    crs_plot: int,
    bbox_wsen: tuple[float, float, float, float] | None,
    log_scale: bool,
    vmax_percentile: float | None,
    full_range: bool,
    figsize_in: tuple[float, float],
    polygon_edges: bool,
) -> int:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np
    except ImportError as e:
        print(f"ERROR: {e}. Install geopandas matplotlib.", file=sys.stderr)
        return 1

    if not path.is_file():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    layer_name = _resolve_layer(path, layer)
    gdf = gpd.read_file(path, layer=layer_name)
    n_in = len(gdf)

    if bbox_wsen is not None:
        west, south, east, north = bbox_wsen
        gdf = _filter_europe_extent(gdf, west, south, east, north)
        print(f"Extent filter (centroid in WGS84 [{west},{south},{east},{north}]): {n_in:,} -> {len(gdf):,} features")
        if len(gdf) == 0:
            print(
                "ERROR: No features left after bbox filter. Widen --bbox or use --no-bbox-filter.",
                file=sys.stderr,
            )
            return 1

    if max_rows is not None and len(gdf) > max_rows:
        gdf = gdf.sample(n=max_rows, random_state=42)
        print(f"Sampled to {len(gdf):,} rows (--max-rows).")

    value_col = field or _infer_value_column(gdf, layer_name)
    if not value_col or value_col not in gdf.columns:
        print(
            f"ERROR: No numeric value column. Columns: {list(gdf.columns)}. Use --field.",
            file=sys.stderr,
        )
        return 1

    value_label = _value_label(layer_name, value_col)

    s = gdf[value_col]
    valid = s.notna() & np.isfinite(s.astype(float))
    if not valid.any():
        print(f"ERROR: Column {value_col!r} has no finite values.", file=sys.stderr)
        return 1

    s_valid = s.loc[valid].astype(float)
    data_min = float(s_valid.min())
    data_max = float(s_valid.max())
    vmin = data_min
    vmax = data_max
    if vmin == vmax:
        vmax = vmin + 1e-9

    norm = None
    legend_kwds: dict = {"shrink": 0.55, "label": value_label}
    capped_note = ""

    if not log_scale and not full_range and vmax_percentile is not None:
        pct = float(vmax_percentile)
        if 0 < pct < 100:
            cap = float(np.percentile(s_valid.to_numpy(), pct))
            if np.isfinite(cap) and cap > vmin:
                vmax = cap
                capped_note = f"{pct:g}th pct"
                print(
                    f"Colour scale: [{vmin:.6g}, {vmax:.6g}] ({capped_note} as vmax); "
                    f"raw range [{data_min:.6g}, {data_max:.6g}] (values above vmax saturate)."
                )

    plot_kwargs: dict = {
        "column": value_col,
        "cmap": cmap,
        "missing_kwds": {"color": "lightgrey", "label": "No data"},
        "legend_kwds": legend_kwds,
        "linewidth": 0.15 if polygon_edges else 0.0,
        "edgecolor": "0.85" if polygon_edges else "none",
    }

    plot_col = value_col
    if log_scale:
        s_float = s.astype(float)
        pos_mask = valid & (s_float > 0)
        if not pos_mask.any():
            print("WARN: --log requested but no positive values; using linear scale.", file=sys.stderr)
        else:
            vpos = s_float.loc[pos_mask]
            lo = max(float(vpos.min()), 1e-12)
            hi = max(float(vpos.max()), lo * (1.0 + 1e-9))
            norm = LogNorm(vmin=lo, vmax=hi)
            legend_kwds["label"] = f"{value_label} (log scale)"
            plot_kwargs["missing_kwds"] = {"color": "lightgrey", "label": "No data or 0"}
            gdf = gdf.copy()
            plot_col = "_map_log_value"
            gdf[plot_col] = s_float
            gdf.loc[s_float <= 0, plot_col] = np.nan
            plot_kwargs["column"] = plot_col

    if norm is None:
        plot_kwargs["vmin"] = vmin
        plot_kwargs["vmax"] = vmax
        if capped_note:
            legend_kwds["label"] = f"{value_label} (vmax={vmax:.4g}, {capped_note})"
    else:
        plot_kwargs["norm"] = norm

    g_plot = gdf.to_crs(crs_plot)

    w, h = figsize_in
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    g_plot.plot(ax=ax, legend=True, **plot_kwargs)
    title_extra = "" if bbox_wsen is None else " (Europe extent)"
    if capped_note and not log_scale:
        title_extra += f"\nvmax={vmax:.4g} ({capped_note}); max cell={data_max:.4g}"
    title_fs = max(10, min(13, 8 + 0.25 * (w + h)))
    ax.set_title(f"{path.name} — {layer_name!r} — {value_label}{title_extra}", fontsize=title_fs)
    if crs_plot == 4326:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.text(
            0.0,
            1.01,
            "EPSG:3035 LAEA Europe",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )
    ax.set_aspect("equal", adjustable="box")

    xmin, ymin, xmax, ymax = g_plot.total_bounds
    pad_x = (xmax - xmin) * 0.02 + 1e3
    pad_y = (ymax - ymin) * 0.02 + 1e3
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Wrote {out.resolve()} (fig {w:g}x{h:g} in, {dpi} dpi)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=_ROOT / "data" / "Agriculture" / "C21.gpkg",
        help="Path to C21.gpkg",
    )
    p.add_argument("--layer", default=None, help="Layer name or index (default: first layer)")
    p.add_argument("--field", default=None, help="Numeric column to color by (default: infer)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: next to gpkg, C21_<layer>_<field>.png)",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Random sample size for faster maps")
    p.add_argument(
        "--dpi",
        type=int,
        default=_DEFAULT_SAVE_DPI,
        help=f"Raster resolution (default {_DEFAULT_SAVE_DPI}). Use 400–600 for print.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=list(_DEFAULT_FIGSIZE_IN),
        help=f"Figure size in inches width height (default {_DEFAULT_FIGSIZE_IN[0]} {_DEFAULT_FIGSIZE_IN[1]})",
    )
    p.add_argument(
        "--polygon-edges",
        action="store_true",
        help="Draw faint polygon outlines (default: no edges, smoother fill)",
    )
    p.add_argument("--cmap", default="YlOrRd")
    p.add_argument(
        "--crs",
        type=int,
        default=3035,
        help="CRS for display (default 3035 LAEA Europe)",
    )
    p.add_argument(
        "--bbox",
        type=str,
        default=None,
        help=f"West,south,east,north in WGS84 for centroid filter (default: {_DEFAULT_EUROPE_BBOX[0]},{_DEFAULT_EUROPE_BBOX[1]},{_DEFAULT_EUROPE_BBOX[2]},{_DEFAULT_EUROPE_BBOX[3]})",
    )
    p.add_argument(
        "--no-bbox-filter",
        action="store_true",
        help="Plot all features (overseas territories may shrink the extent)",
    )
    p.add_argument(
        "--log",
        action="store_true",
        dest="log_scale",
        help="Log color scale (positive values only; 0 / missing shown as grey)",
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.0,
        help="Linear scale: vmax = this percentile of valid values (100 = use true max). Default 99.",
    )
    p.add_argument(
        "--full-range",
        action="store_true",
        help="Linear scale: vmin/vmax = data min/max (ignore percentile cap; can wash out the map)",
    )
    p.add_argument(
        "--batch-counts",
        action="store_true",
        help="Write five PNGs: Bovine, Goats, Pigs (A3100), Poultry, Sheep (counts not density)",
    )
    args = p.parse_args()
    path = args.path.resolve()

    if args.batch_counts:
        if args.layer is not None or args.field is not None or args.out is not None:
            print(
                "ERROR: use --batch-counts alone (do not pass --layer, --field, or --out).",
                file=sys.stderr,
            )
            return 2

    layer_arg: str | int | None = args.layer
    if layer_arg is not None:
        try:
            layer_arg = int(layer_arg)
        except ValueError:
            pass

    if args.no_bbox_filter:
        bbox_wsen = None
    else:
        bbox_wsen = _parse_bbox(args.bbox) if args.bbox is not None else _DEFAULT_EUROPE_BBOX

    if args.batch_counts:
        rc = 0
        for lyr_name, fld in LIVESTOCK_COUNT_LAYERS:
            safe_fld = "".join(c if c.isalnum() else "_" for c in fld)
            out_one = path.parent / f"C21_map_{lyr_name}_{safe_fld}.png"
            r = _plot(
                path,
                lyr_name,
                fld,
                out_one,
                args.max_rows,
                args.dpi,
                args.cmap,
                args.crs,
                bbox_wsen,
                args.log_scale,
                args.vmax_percentile,
                args.full_range,
                (float(args.figsize[0]), float(args.figsize[1])),
                args.polygon_edges,
            )
            if r != 0:
                rc = r
        return rc

    out = args.out
    if out is None:
        lyr = _resolve_layer(path, layer_arg)
        import geopandas as gpd

        gdf = gpd.read_file(path, layer=lyr, rows=1)
        fld = args.field or _infer_value_column(gdf, lyr) or "value"
        safe_lyr = "".join(c if c.isalnum() else "_" for c in lyr)
        safe_fld = "".join(c if c.isalnum() else "_" for c in str(fld))
        out = path.parent / f"C21_map_{safe_lyr}_{safe_fld}.png"

    return _plot(
        path,
        layer_arg,
        args.field,
        out,
        args.max_rows,
        args.dpi,
        args.cmap,
        args.crs,
        bbox_wsen,
        args.log_scale,
        args.vmax_percentile,
        args.full_range,
        (float(args.figsize[0]), float(args.figsize[1])),
        args.polygon_edges,
    )


if __name__ == "__main__":
    raise SystemExit(main())
