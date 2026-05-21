"""
VIIRS active fire archive (FIRMS SV-C2 shapefile) — quick look over Greece.

FIRMS Suomi NPP / NOAA-20 VIIRS active fire detections are ~375 m at nadir.
Points are accumulated into 375 m bins in EPSG:3035 (not 0.25°), then drawn in
WGS84 for display. No warp to the PROXY 100 m reference grid.

Run from repo root::

    python PROXY_V2/dataset_loaders/tests/visualize_viirs_greece.py
    python PROXY_V2/dataset_loaders/tests/visualize_viirs_greece.py --list
    python PROXY_V2/dataset_loaders/tests/visualize_viirs_greece.py --metric frp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DEFAULT_VIIRS_DIR = _REPO / "INPUT/Proxy/ProxySpecific/Agriculture/VIIRS"
_VIIRS_CELL_M = 375.0
_METRIC_CRS = "EPSG:3035"
_GR_LON = (19.0, 30.0)
_GR_LAT = (34.0, 42.0)


def _shp_path(viirs_dir: Path) -> Path:
    hits = sorted(viirs_dir.glob("fire_archive_SV-C2_*.shp"))
    if not hits:
        raise FileNotFoundError(f"No fire_archive_SV-C2_*.shp in {viirs_dir}")
    return hits[0]


def explore_viirs_folder(viirs_dir: Path) -> None:
    print(f"VIIRS folder: {viirs_dir.resolve()}")
    if not viirs_dir.is_dir():
        print("  (missing)")
        return
    for pat in ("fire_archive_SV-C2_*.shp", "fire_archive_SV-C2_*.dbf", "Readme.txt"):
        files = list(viirs_dir.glob(pat))
        print(f"  {pat}: {len(files)} file(s)" + (f"  ({files[0].name})" if len(files) == 1 else ""))
    try:
        shp = _shp_path(viirs_dir)
        import geopandas as gpd

        g = gpd.read_file(shp, rows=1)
        print(f"  CRS: {g.crs}")
        print(f"  Columns: {', '.join(c for c in g.columns if c != 'geometry')}")
        print(f"  Native product resolution: {_VIIRS_CELL_M:.0f} m (VIIRS I-band, FIRMS SV-C2)")
    except FileNotFoundError as e:
        print(f"  {e}")


def load_greece_fires(viirs_dir: Path) -> "object":
    import geopandas as gpd

    shp = _shp_path(viirs_dir)
    g = gpd.read_file(shp, bbox=(_GR_LON[0], _GR_LAT[0], _GR_LON[1], _GR_LAT[1]))
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    elif str(g.crs) != "EPSG:4326":
        g = g.to_crs("EPSG:4326")
    return g


def fires_to_grid(
    gdf: object,
    *,
    metric: str,
    cell_m: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    from rasterio.warp import transform_bounds

    g = gdf.to_crs(_METRIC_CRS)
    x = np.asarray(g.geometry.x, dtype=np.float64)
    y = np.asarray(g.geometry.y, dtype=np.float64)
    if metric == "frp":
        w = np.asarray(g["FRP"], dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    else:
        w = np.ones(len(x), dtype=np.float64)

    cell = float(cell_m)
    xmin, ymin, xmax, ymax = g.total_bounds
    x0 = np.floor(xmin / cell) * cell
    x1 = np.ceil(xmax / cell) * cell
    y0 = np.floor(ymin / cell) * cell
    y1 = np.ceil(ymax / cell) * cell
    x_edges = np.arange(x0, x1 + cell, cell)
    y_edges = np.arange(y0, y1 + cell, cell)

    grid, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=w)
    grid = np.flipud(grid).astype(np.float32)

    west, south, east, north = transform_bounds(
        _METRIC_CRS, "EPSG:4326", float(x_edges[0]), float(y_edges[0]), float(x_edges[-1]), float(y_edges[-1])
    )
    return grid, (west, south, east, north)


def plot_greece(
    field: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    masked = np.ma.masked_where(field <= 0, field)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = field[field > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    im = ax.imshow(
        masked,
        origin="upper",
        extent=extent,
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax if vmax > 0 else None,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label=cbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--viirs-dir", type=Path, default=_DEFAULT_VIIRS_DIR)
    p.add_argument("--list", action="store_true", help="Print folder layout and exit")
    p.add_argument(
        "--metric",
        choices=("count", "frp"),
        default="count",
        help="count = detections per cell; frp = sum FRP (MW) per cell",
    )
    p.add_argument("--cell-m", type=float, default=_VIIRS_CELL_M, help="Bin size in metres (default 375)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO / "PROXY_V2/dataset_loaders/tests/output/viirs_greece_fires.png",
    )
    args = p.parse_args()

    if args.list:
        explore_viirs_folder(args.viirs_dir)
        return

    gdf = load_greece_fires(args.viirs_dir)
    n = len(gdf)
    if n == 0:
        print("No VIIRS detections in Greece bbox")
        sys.exit(1)

    grid, extent = fires_to_grid(gdf, metric=args.metric, cell_m=args.cell_m)
    pos = grid[grid > 0]
    d0 = gdf["ACQ_DATE"].min()
    d1 = gdf["ACQ_DATE"].max()
    print(
        f"Greece ({_GR_LON[0]}–{_GR_LON[1]}°E, {_GR_LAT[0]}–{_GR_LAT[1]}°N): "
        f"{n:,} detections  {d0.date()} .. {d1.date()}  "
        f"bins {args.cell_m:.0f} m ({_METRIC_CRS}) shape={grid.shape}  "
        f"cells>0={int((grid > 0).sum())}  max={float(grid.max()):.6g}"
    )

    cell = args.cell_m
    if args.metric == "frp":
        label = f"sum FRP per {cell:.0f} m cell (MW)"
        title = f"VIIRS active fire — Greece (FRP, {cell:.0f} m)"
    else:
        label = f"detections per {cell:.0f} m cell"
        title = f"VIIRS active fire — Greece (count, {cell:.0f} m)"

    plot_greece(grid, extent, title=title, cbar_label=label, out_path=args.output)


if __name__ == "__main__":
    main()
