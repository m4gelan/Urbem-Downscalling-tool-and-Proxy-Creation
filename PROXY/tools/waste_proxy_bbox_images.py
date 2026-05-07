#!/usr/bin/env python3
"""J_Waste proxy PNG exports for a WGS84 bbox (default: Athens–Attica).

Produces:
  - Solid: CORINE-only map, OSM infrastructure map, plus **combined solid proxy** (normalized stack).
  - Wastewater: four stack layers + **combined wastewater proxy**.
  - Residual: three inputs + **combined residual proxy**.
  - Weights: CH4 and NOx — **per-CAMS-cell** stretch when CAMS NetCDF is available (shows high weights
    inside cells); otherwise 2–98% log percentile. Colorbars on scalar maps; legends on categorical maps.

Optional OSM basemap (EPSG:3857 tiles warped to the view grid) and CAMS GNFR J area cell outlines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default: central Athens / inner Attica (WGS84, degrees)
DEFAULT_BBOX_WGS84 = (23.50, 37.85, 23.95, 38.08)

# Match ``PROXY.visualization.waste_area_map._WASTE_DATASET_COLOR_OVERRIDES`` (hex without #)
_CORINE_121_RGB = (234, 88, 12)
_CORINE_132_RGB = (180, 83, 9)

_OSML_LAYER_RGB: dict[str, tuple[int, int, int]] = {
    "J_Waste · solid: osm_landfill": (217, 119, 6),
    "J_Waste · solid: osm_amenity_recycling": (245, 158, 11),
    "J_Waste · solid: osm_amenity_waste_disposal": (185, 28, 28),
    "J_Waste · solid: osm_wastewater_plant": (220, 38, 38),
}

_WW_KEYS = (
    "J_Waste · WW stack: uwwtd_agglomerations",
    "J_Waste · WW stack: uwwtd_treatment_plants",
    "J_Waste · WW stack: imperviousness",
    "J_Waste · WW stack: population",
)
_RESIDUAL_KEYS = (
    "J_Waste · residual: residual_pop",
    "J_Waste · residual: residual_ghsl_rural_mask",
    "J_Waste · residual: residual_imperv_01",
)

_WW_FILENAMES = {
    "J_Waste · WW stack: uwwtd_agglomerations": "waste_bbox_ww_agglomerations.png",
    "J_Waste · WW stack: uwwtd_treatment_plants": "waste_bbox_ww_treatment_plants.png",
    "J_Waste · WW stack: imperviousness": "waste_bbox_ww_imperviousness.png",
    "J_Waste · WW stack: population": "waste_bbox_ww_population.png",
}
_RESIDUAL_FILENAMES = {
    "J_Waste · residual: residual_pop": "waste_bbox_residual_population.png",
    "J_Waste · residual: residual_ghsl_rural_mask": "waste_bbox_residual_rural.png",
    "J_Waste · residual: residual_imperv_01": "waste_bbox_residual_imperviousness.png",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> Path:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _reproject_basemap_to_overlay_grid(
    img_merc: object,
    extent_merc: tuple[float, float, float, float],
    dst_transform,
    dst_shape: tuple[int, int],
) -> object:
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    img_merc = np.asarray(img_merc)
    left, right, bottom, top = extent_merc
    hm, wm = int(img_merc.shape[0]), int(img_merc.shape[1])
    src_transform = from_bounds(left, bottom, right, top, wm, hm)
    crs3857 = rasterio.crs.CRS.from_epsg(3857)
    crs4326 = rasterio.crs.CRS.from_epsg(4326)
    gray = 0.933
    base_rgb = np.full((gh, gw, 3), gray, dtype=np.float32)
    src_arr = np.asarray(img_merc[..., :3], dtype=np.float32) / 255.0
    for k in range(3):
        dst_band = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=src_arr[..., k].astype(np.float64),
            destination=dst_band,
            src_transform=src_transform,
            src_crs=crs3857,
            dst_transform=dst_transform,
            dst_crs=crs4326,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        valid = np.isfinite(dst_band)
        base_rgb[..., k] = np.where(valid, dst_band.astype(np.float32), gray)
    return base_rgb


def _alpha_composite_rgb_under_rgba(base_rgb: object, overlay_rgba: object) -> object:
    import numpy as np

    base_f = np.asarray(base_rgb, dtype=np.float32)
    # Basemap path uses floats in [0, 1]; chained composites pass uint8 [0, 255].
    if base_f.ndim == 3 and base_f.shape[2] == 3 and float(np.nanmax(base_f)) > 1.5:
        base_f = base_f / 255.0
    base_f = np.clip(base_f, 0.0, 1.0)

    over = np.asarray(overlay_rgba, dtype=np.float32) / 255.0
    alpha = np.clip(over[..., 3:4], 0.0, 1.0)
    rgb_o = over[..., :3]
    out = rgb_o * alpha + base_f * (1.0 - alpha)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _composite_rgba_over_osm(
    rgba: object,
    dst_transform,
    dst_shape: tuple[int, int],
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    zoom_adjust: int | None,
) -> object:
    import contextily as ctx
    import numpy as np

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    if np.asarray(rgba).shape[:2] != (gh, gw):
        raise ValueError(f"RGBA shape mismatch vs ({gh}, {gw})")
    img_merc, extent = ctx.bounds2img(
        west,
        south,
        east,
        north,
        zoom="auto",
        ll=True,
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom_adjust=zoom_adjust,
    )
    base = _reproject_basemap_to_overlay_grid(img_merc, extent, dst_transform, (gh, gw))
    return _alpha_composite_rgb_under_rgba(base, rgba)


def _plot_cams_grid(ax, grid_fc: dict, *, color: str = "#0d47a1", lw: float = 1.35, alpha: float = 0.92) -> None:
    try:
        from shapely.geometry import shape
    except ImportError:
        print("WARNING: shapely missing; skipping CAMS grid.", file=sys.stderr)
        return
    for ft in grid_fc.get("features") or []:
        geom = ft.get("geometry")
        if not geom:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=10, solid_capstyle="round")
        elif g.geom_type == "MultiPolygon":
            for part in g.geoms:
                x, y = part.exterior.xy
                ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=10, solid_capstyle="round")


def _percentile_vmin_vmax(z: object, *, lo_q: float = 2.0, hi_q: float = 98.0) -> tuple[float, float] | None:
    import numpy as np

    a = np.asarray(z, dtype=np.float64)
    m = np.isfinite(a) & (a > 0)
    if not np.any(m):
        return None
    v = a[m]
    lo, hi = float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi


def _save_png(
    img: object,
    *,
    title: str,
    west: float,
    south: float,
    east: float,
    north: float,
    out_path: Path,
    dpi: int,
    grid_fc: dict | None,
    legend_entries: list[tuple[str, tuple[int, int, int]]] | None = None,
    colorbar_spec: dict[str, object] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import numpy as np

    arr = np.asarray(img)
    gh, gw = int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4, got {arr.shape}")
    lat_c = np.radians((south + north) / 2.0)
    lon_e = east - west
    lat_e = north - south
    geo_h_over_w = lat_e / (lon_e * np.cos(lat_c))
    extra_w = 1.15 if (colorbar_spec or legend_entries) else 0.35
    fig_w = max(6.0, gw / dpi) + extra_w
    fig_h = max(4.5, fig_w * geo_h_over_w + 0.85)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / np.cos(lat_c), adjustable="box")
    interp = "bilinear" if arr.shape[2] == 3 else "nearest"
    ax.imshow(arr, extent=[west, east, south, north], origin="upper", interpolation=interp, zorder=1)
    if grid_fc:
        _plot_cams_grid(ax, grid_fc)
    ax.set_xlabel("Longitude (deg E)", fontsize=9)
    ax.set_ylabel("Latitude (deg N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(True, color="white", linestyle="--", linewidth=0.3, alpha=0.4, zorder=5)

    if legend_entries:
        handles = [
            mpatches.Patch(
                facecolor=tuple(c / 255.0 for c in rgb),
                edgecolor="0.2",
                linewidth=0.4,
                label=lab,
            )
            for lab, rgb in legend_entries
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.92)

    if colorbar_spec is not None:
        vmin = float(colorbar_spec["vmin"])
        vmax = float(colorbar_spec["vmax"])
        cmap = str(colorbar_spec.get("cmap", "viridis"))
        clab = str(colorbar_spec.get("label", ""))
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.92)
        if colorbar_spec.get("percent_ticks"):
            from matplotlib.ticker import PercentFormatter

            cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if clab:
            cb.set_label(clab, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _rgba_solid_corine(
    scalars: dict[str, object], shape: tuple[int, int]
) -> tuple[object | None, list[tuple[str, tuple[int, int, int]]] | None]:
    """CLC 121 vs 132 on one map (132 draws over 121 where both)."""
    import numpy as np

    gh, gw = shape
    k121 = "J_Waste · solid: corine_clc_121"
    k132 = "J_Waste · solid: corine_clc_132"
    if k121 not in scalars or k132 not in scalars:
        return None, None
    g121 = np.asarray(scalars[k121], dtype=np.float64)
    g132 = np.asarray(scalars[k132], dtype=np.float64)
    m121 = np.isfinite(g121) & (g121 > 0.5)
    m132 = np.isfinite(g132) & (g132 > 0.5)
    rgba = np.zeros((gh, gw, 4), dtype=np.uint8)
    rgba[m121] = (*_CORINE_121_RGB, 255)
    rgba[m132] = (*_CORINE_132_RGB, 255)
    if not np.any(rgba[..., 3] > 0):
        return None, None
    legend = [
        ("CLC 121 — industrial / commercial", _CORINE_121_RGB),
        ("CLC 132 — dump sites", _CORINE_132_RGB),
    ]
    return rgba, legend


def _rgba_solid_osm_stack(
    scalars: dict[str, object], shape: tuple[int, int]
) -> tuple[object | None, list[tuple[str, tuple[int, int, int]]] | None]:
    """All solid OSM context masks with distinct colours (stable key order; later keys paint over earlier)."""
    import numpy as np

    gh, gw = shape
    keys = sorted(k for k in scalars if k.startswith("J_Waste · solid: osm_"))
    if not keys:
        return None, None
    cat = np.zeros((gh, gw), dtype=np.int32)
    for i, key in enumerate(keys, start=1):
        arr = np.asarray(scalars[key], dtype=np.float64)
        m = np.isfinite(arr) & (arr > 0.5)
        cat[m] = i
    if not np.any(cat > 0):
        return None, None
    rgba = np.zeros((gh, gw, 4), dtype=np.uint8)
    legend: list[tuple[str, tuple[int, int, int]]] = []
    for i, key in enumerate(keys, start=1):
        rgb = _OSML_LAYER_RGB.get(key)
        if rgb is None:
            h = (37 * i) % 200 + 40
            rgb = (h, (13 * i + 80) % 200, (91 * i) % 180 + 50)
        short = key.replace("J_Waste · solid: osm_", "").replace("_", " ")
        legend.append((short, rgb))
        m = cat == i
        rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = rgb[0], rgb[1], rgb[2], 255
    return rgba, legend


def main() -> int:
    root = _ensure_import_path()
    ap = argparse.ArgumentParser(
        description="Export J_Waste proxy maps (solid / WW / residual layers + CH4/NOx weights) for a bbox."
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help=f"WGS84 bbox west south east north (default: Athens {DEFAULT_BBOX_WGS84}).",
    )
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--paths-yaml", type=Path, default=None)
    ap.add_argument("--sector-yaml", type=Path, default=None)
    ap.add_argument(
        "--weight-tif",
        type=Path,
        default=None,
        help="J_Waste area weights GeoTIFF (default: OUTPUT/Proxy_weights/J_Waste/waste_areasource.tif).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--country", default="EL")
    ap.add_argument("--max-width", type=int, default=1400)
    ap.add_argument("--max-height", type=int, default=1200)
    ap.add_argument("--pad-deg", type=float, default=0.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--cams-nc",
        type=Path,
        default=None,
        help="CAMS NetCDF (default: emissions.cams_2019_nc from paths.yaml).",
    )
    ap.add_argument("--skip-cams-grid", action="store_true")
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--basemap-zoom-adjust", type=int, default=None)
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox is not None else DEFAULT_BBOX_WGS84
    west, south, east, north = (float(x) for x in bbox)
    if west >= east or south >= north:
        print("ERROR: require west < east and south < north.", file=sys.stderr)
        return 1

    paths_yaml = args.paths_yaml or (args.root / "PROXY" / "config" / "paths.yaml")
    sector_yaml = args.sector_yaml or (args.root / "PROXY" / "config" / "sectors" / "waste.yaml")
    wt_default = args.root / "OUTPUT" / "Proxy_weights" / "J_Waste" / "waste_areasource.tif"
    weight_tif = args.weight_tif or wt_default
    wt = weight_tif if weight_tif.is_absolute() else args.root / weight_tif

    for label, pth in [
        ("paths.yaml", paths_yaml),
        ("sector YAML", sector_yaml),
        ("weight GeoTIFF", wt),
    ]:
        if not pth.is_file():
            print(f"ERROR: {label} not found: {pth}", file=sys.stderr)
            return 1

    import yaml

    with paths_yaml.open(encoding="utf-8") as f:
        path_cfg = yaml.safe_load(f)
    with sector_yaml.open(encoding="utf-8") as f:
        sector_cfg = yaml.safe_load(f)
    if not isinstance(path_cfg, dict) or not isinstance(sector_cfg, dict):
        print("ERROR: YAML must parse to mappings.", file=sys.stderr)
        return 1

    out_dir = (args.out_dir.resolve() if args.out_dir else Path.cwd().resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    from PROXY.core.dataloaders import resolve_path
    from PROXY.core.dataloaders.discovery import discover_cams_emissions
    from PROXY.core.cams.mask import cams_gnfr_country_source_mask
    from PROXY.sectors.J_Waste.pipeline import merge_waste_pipeline_cfg
    from PROXY.visualization._mapbuilder import (
        build_cams_area_grid_geojson_for_view,
        compute_view_context,
        pick_band_by_pollutant,
        pick_first_positive_band,
        resolve_under_root,
        weight_rgba_percentile,
        weight_rgba_per_cell,
    )
    from PROXY.visualization.overlay_utils import read_weight_wgs84_only
    from PROXY.visualization.waste_context import build_waste_proxy_rgba_overlays
    import numpy as np
    import xarray as xr

    wt_resolved = resolve_under_root(wt, args.root)
    waste_merged = merge_waste_pipeline_cfg(
        args.root,
        path_cfg,
        sector_cfg,
        country=str(args.country),
        output_dir=wt_resolved.parent.resolve(),
    )

    view = compute_view_context(
        wt_resolved,
        pad_deg=float(args.pad_deg),
        max_width=int(args.max_width),
        max_height=int(args.max_height),
        override_bbox=(west, south, east, north),
    )

    scalars: dict[str, object] = {}
    overlays = build_waste_proxy_rgba_overlays(
        args.root,
        waste_merged,
        wt_resolved,
        view.west,
        view.south,
        view.east,
        view.north,
        view.dst_t,
        (view.gh, view.gw),
        path_cfg,
        resampling="bilinear",
        scalars_out=scalars,
    )
    by_title = {t: rgba for t, _, rgba in overlays}
    title_to_cmap = {t: cm for t, cm, _ in overlays}

    grid_fc: dict | None = None
    cams_ds: xr.Dataset | None = None
    m_area: object | None = None
    nc_path_resolved: Path | None = None
    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    cams_block = sector_cfg.get("cams") or {}
    gnfr = str(cams_block.get("gnfr", "J"))
    domain_bbox = cams_block.get("domain_bbox_wgs84")
    domain_bbox_t = tuple(float(x) for x in domain_bbox) if domain_bbox else None
    stypes = tuple(cams_block.get("source_types") or ("area",))

    if not args.skip_cams_grid:
        em = path_cfg.get("emissions") or {}
        nc_rel = em.get("cams_2019_nc")
        nc_path = None
        if args.cams_nc is not None:
            nc_path = args.cams_nc if args.cams_nc.is_absolute() else args.root / args.cams_nc
        elif nc_rel:
            nc_path = discover_cams_emissions(args.root, resolve_path(args.root, Path(str(nc_rel))))
            nc_path = nc_path if nc_path.is_absolute() else args.root / nc_path
        if nc_path is not None and nc_path.is_file():
            nc_path_resolved = nc_path
            cams_ds = xr.open_dataset(nc_path, engine="netcdf4")
            try:
                m_area = cams_gnfr_country_source_mask(
                    cams_ds,
                    iso3,
                    gnfr=gnfr,
                    source_types=stypes,
                    domain_bbox_wgs84=domain_bbox_t,
                )
                grid_fc = build_cams_area_grid_geojson_for_view(cams_ds, m_area, view)
            except Exception as exc:
                print(f"WARNING: CAMS grid / mask failed ({exc}).", file=sys.stderr)
                cams_ds.close()
                cams_ds = None
                m_area = None
                grid_fc = None
                nc_path_resolved = None

    try:
        import contextily  # noqa: F401
    except ImportError:
        contextily = None
    use_basemap = (not args.no_basemap) and (contextily is not None)
    if not args.no_basemap and contextily is None:
        print("WARNING: install contextily for OSM basemap.", file=sys.stderr)

    def _maybe_basemap(rgba: object) -> object:
        if not use_basemap:
            return rgba
        try:
            return _composite_rgba_over_osm(
                rgba,
                view.dst_t,
                (view.gh, view.gw),
                view.west,
                view.south,
                view.east,
                view.north,
                zoom_adjust=args.basemap_zoom_adjust,
            )
        except Exception as exc:
            print(f"  WARNING: basemap failed ({exc}).", file=sys.stderr)
            return rgba

    shp = (view.gh, view.gw)
    jobs: list[
        tuple[str, str, object, list[tuple[str, tuple[int, int, int]]] | None, dict[str, object] | None]
    ] = []

    rc, leg_c = _rgba_solid_corine(scalars, shp)
    if rc is not None:
        jobs.append(("J_Waste — solid: CORINE (121 / 132)", "waste_bbox_solid_corine.png", rc, leg_c, None))
    ro, leg_o = _rgba_solid_osm_stack(scalars, shp)
    if ro is not None:
        jobs.append(("J_Waste — solid: OSM infrastructure", "waste_bbox_solid_osm.png", ro, leg_o, None))

    _COMBINED = (
        ("J_Waste · proxy: solid (combined, normalized)", "waste_bbox_proxy_solid_combined.png", "plasma"),
        ("J_Waste · proxy: wastewater (combined, normalized)", "waste_bbox_proxy_wastewater_combined.png", "inferno"),
        ("J_Waste · proxy: residual (combined, normalized)", "waste_bbox_proxy_residual_combined.png", "cividis"),
    )
    for title_c, fn_c, cmap_c in _COMBINED:
        if title_c in by_title:
            jobs.append(
                (
                    "J_Waste — combined proxy: "
                    + title_c.replace("J_Waste · proxy: ", "").replace(" (combined, normalized)", ""),
                    fn_c,
                    by_title[title_c],
                    None,
                    {"vmin": 0.0, "vmax": 1.0, "cmap": cmap_c, "label": "Combined proxy (0–1)"},
                )
            )

    for key in _WW_KEYS:
        fn = _WW_FILENAMES.get(key)
        if fn and key in by_title:
            short = key.replace("J_Waste · WW stack: ", "")
            cm = title_to_cmap.get(key, "viridis")
            z = scalars.get(key)
            cb: dict[str, object] | None = None
            if z is not None:
                pv = _percentile_vmin_vmax(z)
                if pv:
                    lo, hi = pv
                    cb = {"vmin": lo, "vmax": hi, "cmap": cm, "label": short.replace("_", " ")}
            jobs.append((f"J_Waste — WW: {short}", fn, by_title[key], None, cb))

    for key in _RESIDUAL_KEYS:
        fn = _RESIDUAL_FILENAMES.get(key)
        if fn and key in by_title:
            short = key.replace("J_Waste · residual: ", "")
            cm = title_to_cmap.get(key, "viridis")
            z = scalars.get(key)
            cb2: dict[str, object] | None = None
            if z is not None:
                pv = _percentile_vmin_vmax(z)
                if pv:
                    lo, hi = pv
                    cb2 = {"vmin": lo, "vmax": hi, "cmap": cm, "label": short.replace("_", " ")}
            jobs.append((f"J_Waste — residual: {short}", fn, by_title[key], None, cb2))

    viz_cfg = sector_cfg.get("visualization") or {}
    strip = ("j_waste_weight_",)

    def _weight_rgba_and_cbar(pol: str) -> tuple[object, int, dict[str, object] | None] | None:
        band = pick_band_by_pollutant(
            wt_resolved,
            {**viz_cfg, "visualization_pollutant": pol},
            strip_prefixes=strip,
        )
        band, _ = pick_first_positive_band(
            wt_resolved,
            band,
            empty_message=f"No positive weights for {pol}; using band anyway.",
        )
        stk = read_weight_wgs84_only(
            wt_resolved,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            display_width=view.gw,
            display_height=view.gh,
            weight_band=int(band),
        )
        w_arr = stk["weight_wgs84"]
        w_nd = stk["weight_nodata"]
        if cams_ds is not None and m_area is not None and nc_path_resolved is not None:
            rgba = weight_rgba_per_cell(
                w_arr,
                w_nodata=w_nd,
                cams_nc_path=nc_path_resolved,
                m_area=m_area,
                ds=cams_ds,
                view=view,
                cmap="plasma",
            )
            cbar_d: dict[str, object] = {
                "vmin": 0.0,
                "vmax": 1.0,
                "cmap": "plasma",
                "label": f"{pol} weight (per CAMS cell, 0–1)",
            }
        else:
            rgba = weight_rgba_percentile(w_arr, w_nodata=w_nd, cmap="plasma")
            pv = _percentile_vmin_vmax(w_arr)
            if pv:
                lo, hi = pv
                cbar_d = {
                    "vmin": lo,
                    "vmax": hi,
                    "cmap": "plasma",
                    "label": f"{pol} weight (2–98% tile)",
                }
            else:
                cbar_d = {"vmin": 0.0, "vmax": 1.0, "cmap": "plasma", "label": f"{pol} weight"}
        if not np.any(rgba[..., 3] > 0):
            return None
        return rgba, int(band), cbar_d

    for pol, fn, title_suffix in (
        ("CH4", "waste_bbox_weights_ch4.png", "CH4"),
        ("NOx", "waste_bbox_weights_nox.png", "NOx"),
        ("NMVOC", "waste_bbox_weights_nmvoc.png", "NMVOC"),
    ):
        got = _weight_rgba_and_cbar(pol)
        if got is None:
            print(f"WARNING: no weight raster for {pol}; skip {fn}.", file=sys.stderr)
            continue
        rgba_w, bd, cbar_d = got
        jobs.append(
            (
                f"J_Waste — weights ({title_suffix}, band {bd})",
                fn,
                rgba_w,
                None,
                cbar_d,
            )
        )

    if cams_ds is not None:
        cams_ds.close()
        cams_ds = None

    print(
        f"View {view.gw}x{view.gh} | bbox [{view.west:.4f}, {view.south:.4f}, {view.east:.4f}, {view.north:.4f}]"
    )
    for title, fname, rgba, legend_entries, colorbar_spec in jobs:
        print(f"\n{title}")
        out = _maybe_basemap(rgba)
        _save_png(
            out,
            title=title,
            west=view.west,
            south=view.south,
            east=view.east,
            north=view.north,
            out_path=out_dir / fname,
            dpi=int(args.dpi),
            grid_fc=grid_fc,
            legend_entries=legend_entries,
            colorbar_spec=colorbar_spec,
        )

    if use_basemap:
        print(
            "\nBasemap: OpenStreetMap – https://www.openstreetmap.org/copyright",
            file=sys.stderr,
        )
    print(f"\nDone. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
