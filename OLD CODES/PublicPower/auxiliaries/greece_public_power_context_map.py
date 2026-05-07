#!/usr/bin/env python3
"""
Folium map for Greece (mainland NUTS2 union): stacked gridded layers plus CAMS / JRC.

Basemaps (layer control): light CartoDB Positron and Esri World Imagery satellite.

Layers (toggle in layer control):
  - LandScan 2020 population (raster, log-scaled, EPSG:4326 grid)
  - CORINE industrial/commercial land cover (see encoding note below)
  - Mainland outline (NUTS2 dissolve)
  - CAMS-REG GNFR A area rectangles + point markers + JRC OPEN UNITS (same as cams script)

CORINE / class 121 (recheck):
  - **CLC code 121** is correct for *Industrial or commercial units and public facilities* (Copernicus
    nomenclature 1.2.1, see land.copernicus.eu CLC2018 guidelines). That is the value stored in
    **single-band classification** GeoTIFFs from EEA.
  - **204, 77, 242, 255** in QGIS/Copernicus legends is usually **RGBA display colour** for class 121,
    not an alternative class ID. If your file is an **RGB (or RGBA) rendered export**, pixels hold
    colour bytes, not 121: use ``--corine-mode rgb`` or default ``both`` to also match legend RGB.
  - UrbEm ``corine_classes.json`` uses internal raster value **3** for industry on some pipelines;
    keep ``--corine-class-codes 121,3`` if needed.

Mainland = Greece (EL) NUTS2 regions excluding island NUTS2 by default:
  EL41, EL42, EL43, EL62 (North Aegean, South Aegean, Crete, Ionian Islands).

Requires: folium, rasterio, geopandas, shapely, matplotlib, numpy, pandas, xarray, netCDF4

Usage (from project root):
  python PublicPower/Auxiliaries/greece_public_power_context_map.py
  python PublicPower/Auxiliaries/greece_public_power_context_map.py --corine Input/CORINE/U2018_CLC2018_V2020_20u1.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Island NUTS2 (Eurostat 2021) excluded from "mainland" mask
DEFAULT_EXCLUDE_NUTS2 = frozenset({"EL41", "EL42", "EL43", "EL62"})

# CLC2018 level-2: 121 = Industrial or commercial units and public facilities (official code).
# Copernicus / QGIS legend colour for 121 (not a class code in the grid — use for RGB GeoTIFFs).
DEFAULT_CLC121_LEGEND_RGB: tuple[int, int, int] = (204, 77, 242)
# UrbEm corine_classes reclass uses 3 for industry on some rasters.
DEFAULT_CORINE_INDUSTRY_CODES: tuple[int, ...] = (121, 3)

DEFAULT_LANDSCAN = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "PublicPower"
    / "landscan"
    / "landscan-global-2020.tif"
)

CORINE_CANDIDATES = [
    Path("Input/CORINE/U2018_CLC2018_V2020_20u1.tif"),
    Path("Input/CORINE/U2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
    Path("data/CORINE/U2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
]

CORINE_LEGEND_CSV_CANDIDATES = [
    Path("data/CORINE/U2018_clc2018_v2020_20u1_fgdb/Legend/CLC_legend.csv"),
]

NUTS_CANDIDATES = [
    Path("Data/geometry/NUTS_RG_20M_2021_3035.gpkg"),
    Path("data/geometry/NUTS_RG_20M_2021_3035.gpkg"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _first_existing(root: Path, candidates: list[Path]) -> Path | None:
    for rel in candidates:
        p = root / rel if not rel.is_absolute() else rel
        if p.is_file():
            return p
    return None


def load_greece_mainland_wgs84(
    nuts_gpkg: Path,
    exclude_nuts2: frozenset[str],
) -> "gpd.GeoDataFrame":
    import geopandas as gpd

    nuts = gpd.read_file(nuts_gpkg)
    el2 = nuts[
        (nuts["CNTR_CODE"].astype(str).str.upper() == "EL") & (nuts["LEVL_CODE"] == 2)
    ]
    if el2.empty:
        raise SystemExit("No Greece (EL) NUTS2 features in NUTS geopackage.")
    nid = el2["NUTS_ID"].astype(str)
    keep = el2[~nid.isin(exclude_nuts2)]
    if keep.empty:
        raise SystemExit("Mainland NUTS2 filter removed all regions; check --exclude-nuts2.")
    dissolved = keep.dissolve()
    return dissolved.to_crs(4326)


def _expand_bounds(
    left: float, bottom: float, right: float, top: float, pad_deg: float
) -> tuple[float, float, float, float]:
    return (
        left - pad_deg,
        bottom - pad_deg,
        right + pad_deg,
        top + pad_deg,
    )


def _rasterize_mask(
    geom_wgs,
    transform,
    height: int,
    width: int,
) -> np.ndarray:
    from rasterio import features

    shapes = [(geom_wgs, 1)]
    return features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


def _reproject_band_to_wgs84_grid(
    raster_path: Path,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
    dst_nodata: float = np.nan,
    band: int = 1,
) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds

    height, width = dst_shape
    dst = np.full((height, width), dst_nodata, dtype=np.float64)
    bounds_wgs84 = rasterio.transform.array_bounds(height, width, dst_transform)

    with rasterio.open(raster_path) as src:
        if band < 1 or band > int(src.count):
            raise SystemExit(
                f"{raster_path.name}: band {band} invalid (file has {src.count} bands)."
            )
        src_crs = src.crs
        if src_crs is None:
            raise SystemExit(f"Raster has no CRS: {raster_path}")
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_crs, *bounds_wgs84, densify_pts=21
        )
        from rasterio.windows import from_bounds

        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return dst
        arr = src.read(band, window=win).astype(np.float64)
        wt = src.window_transform(win)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        res = getattr(Resampling, resampling)
        reproject(
            source=arr,
            destination=dst,
            src_transform=wt,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=res,
        )
    return dst


def _reproject_three_bands_to_wgs84_grid(
    raster_path: Path,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    bands: tuple[int, int, int] = (1, 2, 3),
    resampling: str = "nearest",
) -> np.ndarray:
    """Warp RGB (bands 1–3) to WGS84 grid; shape (height, width, 3) float64."""
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds

    height, width = dst_shape
    bounds_wgs84 = rasterio.transform.array_bounds(height, width, dst_transform)
    out = np.full((height, width, 3), np.nan, dtype=np.float64)

    with rasterio.open(raster_path) as src:
        for i, band in enumerate(bands):
            if band < 1 or band > int(src.count):
                raise SystemExit(
                    f"{raster_path.name}: RGB needs band {band}; file has {src.count} bands."
                )
        src_crs = src.crs
        if src_crs is None:
            raise SystemExit(f"Raster has no CRS: {raster_path}")
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src_crs, *bounds_wgs84, densify_pts=21
        )
        from rasterio.windows import from_bounds

        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return out
        wt = src.window_transform(win)
        res = getattr(Resampling, resampling)
        nodata = src.nodata
        for i, band in enumerate(bands):
            arr = src.read(band, window=win).astype(np.float64)
            if nodata is not None:
                arr = np.where(arr == float(nodata), np.nan, arr)
            plane = np.full((height, width), np.nan, dtype=np.float64)
            reproject(
                source=arr,
                destination=plane,
                src_transform=wt,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=res,
            )
            out[:, :, i] = plane
    return out


def population_to_rgba(
    pop: np.ndarray,
    mask: np.ndarray,
    *,
    vmin: float = 1.0,
    vmax: float | None = None,
) -> np.ndarray:
    import matplotlib.colors as mcolors

    try:
        from matplotlib import colormaps

        cmap = colormaps["viridis"]
    except Exception:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("viridis")

    x = np.array(pop, dtype=np.float64)
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    if vmax is None:
        pos = x[np.isfinite(x) & (mask > 0)]
        vmax = float(np.nanpercentile(pos, 99.0)) if pos.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 10.0
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    h, w = x.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(x) & (x > 0) & (mask > 0)
    rgba[~valid] = [0, 0, 0, 0]
    if np.any(valid):
        c = cmap(norm(x[valid]))
        rgba[valid, 0] = (c[:, 0] * 255).astype(np.uint8)
        rgba[valid, 1] = (c[:, 1] * 255).astype(np.uint8)
        rgba[valid, 2] = (c[:, 2] * 255).astype(np.uint8)
        rgba[valid, 3] = (c[:, 3] * 220).astype(np.uint8)
    return rgba


def industry_corine_mask_from_class(
    clc: np.ndarray,
    mask: np.ndarray,
    *,
    class_codes: tuple[int, ...],
) -> np.ndarray:
    h, w = clc.shape
    u = np.zeros((h, w), dtype=bool)
    ok = np.isfinite(clc) & (mask > 0)
    if not np.any(ok) or not class_codes:
        return u
    rounded = np.zeros((h, w), dtype=np.int32)
    rounded[ok] = np.rint(clc[ok]).astype(np.int32)
    code_set = frozenset(int(c) for c in class_codes)
    for c in code_set:
        u |= ok & (rounded == c)
    return u


def industry_corine_mask_from_legend_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    target_rgb: tuple[int, int, int],
    tolerance: int,
) -> np.ndarray:
    """Match pixels whose R,G,B are near Copernicus legend colour (styled RGB GeoTIFF)."""
    if rgb.shape[-1] != 3:
        raise ValueError("rgb must be (H,W,3)")
    tr, tg, tb = (float(target_rgb[0]), float(target_rgb[1]), float(target_rgb[2]))
    tol = float(tolerance)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    ok = (
        np.isfinite(r)
        & np.isfinite(g)
        & np.isfinite(b)
        & (mask > 0)
        & (np.abs(r - tr) <= tol)
        & (np.abs(g - tg) <= tol)
        & (np.abs(b - tb) <= tol)
    )
    return ok


def industry_mask_to_rgba(u: np.ndarray) -> np.ndarray:
    h, w = u.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[u] = [200, 60, 200, 200]
    return rgba


def industry_corine_to_rgba(
    clc: np.ndarray,
    mask: np.ndarray,
    *,
    class_codes: tuple[int, ...],
) -> np.ndarray:
    return industry_mask_to_rgba(industry_corine_mask_from_class(clc, mask, class_codes=class_codes))


def log_corine_mainland_class_histogram(
    clc: np.ndarray,
    mask: np.ndarray,
    *,
    corine_path: Path,
    class_codes: tuple[int, ...],
) -> None:
    """Print which pixel values occur on mainland after warp (stderr)."""
    ok = np.isfinite(clc) & (mask > 0)
    n_ok = int(np.count_nonzero(ok))
    print(f"CORINE diagnostics ({corine_path.name}):", file=sys.stderr)
    if n_ok == 0:
        print("  No finite pixels inside mainland mask (check CRS, extent, nodata).", file=sys.stderr)
        return
    subs = np.rint(clc[ok]).astype(np.int64)
    uniq, cnt = np.unique(subs, return_counts=True)
    order = np.argsort(-cnt)
    print(f"  Mainland grid cells with data: {n_ok}", file=sys.stderr)
    print("  Top 25 class values by pixel count (after nearest-neighbour warp):", file=sys.stderr)
    for i in order[:25]:
        v, n = int(uniq[i]), int(cnt[i])
        mark = " <- target" if v in frozenset(class_codes) else ""
        print(f"    value {v}: {n} px{mark}", file=sys.stderr)
    rnd = np.zeros_like(clc, dtype=np.int64)
    rnd[ok] = np.rint(clc[ok]).astype(np.int64)
    matched = int(np.count_nonzero(ok & np.isin(rnd, list(class_codes))))
    print(
        f"  Pixels matching --corine-class-codes {','.join(str(c) for c in class_codes)}: {matched}",
        file=sys.stderr,
    )


def _parse_corine_class_codes(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _parse_rgb_triplet(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit(f"--corine-legend-rgb expects R,G,B (got {s!r})")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _parse_legend_rgb_dash_field(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split("-") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Bad legend RGB field {s!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def load_clc_legend_table(
    legend_csv: Path,
) -> tuple[dict[int, int], dict[int, tuple[int, int, int]], dict[int, str]]:
    """CLC_CODE -> GRID_CODE (ordinal in some rasters), RGB, short label."""
    import csv

    clc_to_grid: dict[int, int] = {}
    clc_to_rgb: dict[int, tuple[int, int, int]] = {}
    clc_to_label: dict[int, str] = {}
    with legend_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            raw_clc = (row.get("CLC_CODE") or "").strip()
            raw_grid = (row.get("GRID_CODE") or "").strip()
            if not raw_clc or not raw_grid:
                continue
            try:
                clc = int(raw_clc)
                grid = int(raw_grid)
            except ValueError:
                continue
            clc_to_grid[clc] = grid
            rgb_s = (row.get("RGB") or "").strip()
            if rgb_s:
                try:
                    clc_to_rgb[clc] = _parse_legend_rgb_dash_field(rgb_s)
                except ValueError:
                    pass
            lab = (row.get("LABEL3") or "").strip()
            if lab:
                clc_to_label[clc] = lab
    return clc_to_grid, clc_to_rgb, clc_to_label


def _raster_values_to_polygonize_for_clc(
    clc: int,
    clc_to_grid: dict[int, int],
) -> frozenset[int]:
    vals = {int(clc)}
    g = clc_to_grid.get(int(clc))
    if g is not None:
        vals.add(int(g))
    return frozenset(vals)


def polygonize_corine_classes_in_extent(
    raster_path: Path,
    mainland_wgs: "gpd.GeoDataFrame",
    bounds_wgs84: tuple[float, float, float, float],
    *,
    band: int,
    selected_clcs: tuple[int, ...],
    clc_to_grid: dict[int, int],
    simplify_m: float,
) -> "gpd.GeoDataFrame | None":
    """Extract selected CLC-like classes from the classification band; clip to mainland."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import shape

    left, bottom, right, top = bounds_wgs84
    rows: list[dict] = []
    with rasterio.open(raster_path) as src:
        if band < 1 or band > int(src.count):
            raise SystemExit(
                f"{raster_path.name}: CORINE vector band {band} invalid (file has {src.count})."
            )
        src_crs = src.crs
        if src_crs is None:
            raise SystemExit(f"Raster has no CRS: {raster_path}")
        sl, sb, sr, st = transform_bounds(
            "EPSG:4326", src_crs, left, bottom, right, top, densify_pts=21
        )
        win = from_bounds(sl, sb, sr, st, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return None
        arr = src.read(band, window=win)
        wt = src.window_transform(win)
        nodata = src.nodata
        a = arr.astype(np.float64)
        if nodata is not None:
            a = np.where(a == float(nodata), np.nan, a)
        ok = np.isfinite(a)
        rint = np.zeros_like(a, dtype=np.int32)
        rint[ok] = np.rint(a[ok]).astype(np.int32)

        for clc in selected_clcs:
            vals = _raster_values_to_polygonize_for_clc(int(clc), clc_to_grid)
            mask = np.zeros(a.shape, dtype=bool)
            for v in vals:
                mask |= ok & (rint == int(v))
            if not np.any(mask):
                continue
            bin_img = np.where(mask, np.uint8(1), np.uint8(0)).astype(np.uint8)
            for geom, val in shapes(bin_img, transform=wt, connectivity=8):
                if int(val) != 1:
                    continue
                g = shape(geom)
                if g.is_empty:
                    continue
                rows.append({"clc": int(clc), "geometry": g})

        if not rows:
            return None
        gdf = gpd.GeoDataFrame(rows, crs=src_crs)
        mainland_native = mainland_wgs.to_crs(gdf.crs)
        gdf = gpd.clip(gdf, mainland_native)
        if gdf.empty:
            return None
        if simplify_m and simplify_m > 0:
            gdf = gdf.to_crs(3035)
            gdf["geometry"] = gdf.geometry.simplify(float(simplify_m))
            gdf = gdf.to_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        gdf = gdf[~gdf.geometry.is_empty]
        return gdf if not gdf.empty else None


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(
        description="Greece mainland map: LandScan, CORINE industry classes, CAMS A, JRC."
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=root / "PublicPower" / "outputs" / "greece_public_power_context_map.html",
    )
    p.add_argument("--nuts-gpkg", type=Path, default=None, help="NUTS regions geopackage")
    p.add_argument(
        "--exclude-nuts2",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDE_NUTS2)),
        help="Comma-separated NUTS2 codes to drop (islands)",
    )
    p.add_argument("--pad-deg", type=float, default=0.12, help="Pad around mainland bounds")
    p.add_argument("--grid-width", type=int, default=1400, help="Raster overlay width (px)")
    p.add_argument("--grid-height", type=int, default=1100, help="Raster overlay height (px)")
    p.add_argument("--corine", type=Path, default=None, help="CORINE GeoTIFF path")
    p.add_argument(
        "--corine-mode",
        choices=("class", "rgb", "both"),
        default="both",
        help="class=CLC pixel codes only; rgb=legend colour on bands 1–3; both=union (recommended)",
    )
    p.add_argument(
        "--corine-class-codes",
        type=str,
        default="121,3",
        help="Comma-separated class pixel values: 121=official CLC2018 L2 industry, 3=UrbEm reclass",
    )
    p.add_argument(
        "--corine-legend-rgb",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CLC121_LEGEND_RGB),
        help="Copernicus/QGIS legend R,G,B for class 121 (styled RGB GeoTIFF), not a class code",
    )
    p.add_argument(
        "--corine-rgb-tolerance",
        type=int,
        default=25,
        help="Max difference per channel when matching --corine-legend-rgb",
    )
    p.add_argument(
        "--corine-band",
        type=int,
        default=1,
        help="GeoTIFF band index (1-based) for classification grid (class / both modes)",
    )
    p.add_argument(
        "--corine-vector-classes",
        type=str,
        default="",
        help=(
            "Comma-separated CLC codes to draw as vector polygons (polygonized from the "
            "classification raster in --corine-band). Example: 111,112,121,141. Empty disables."
        ),
    )
    p.add_argument(
        "--corine-vector-legend",
        type=Path,
        default=None,
        help="CLC_legend.csv (GRID_CODE;CLC_CODE;...); autodetect under data/CORINE if omitted",
    )
    p.add_argument(
        "--corine-vector-simplify-m",
        type=float,
        default=200.0,
        help="Simplify vectors in EPSG:3035 (metres); use 0 to disable",
    )
    p.add_argument(
        "--quiet-corine-diagnostics",
        action="store_true",
        help="Do not print mainland CORINE value histogram to stderr",
    )
    p.add_argument("--landscan", type=Path, default=DEFAULT_LANDSCAN, help="LandScan GeoTIFF")
    p.add_argument("--no-landscan", action="store_true")
    p.add_argument("--no-corine", action="store_true")
    p.add_argument("--pollutant", default="nox", help="CAMS pollutant for symbology")
    p.add_argument("--map-max-area-cells", type=int, default=0, help="0 = all Greece A area cells")
    p.add_argument("--no-jrc", action="store_true")
    args = p.parse_args()

    try:
        import folium
        import geopandas as gpd
        import rasterio
        import xarray as xr
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise SystemExit(
            "Need folium, geopandas, rasterio, xarray. "
            "Example: pip install folium geopandas rasterio xarray netCDF4 matplotlib"
        ) from exc

    exclude = frozenset(s.strip().upper() for s in args.exclude_nuts2.split(",") if s.strip())
    vec_raw = (args.corine_vector_classes or "").strip()
    vec_clcs: tuple[int, ...] = (
        _parse_corine_class_codes(vec_raw) if vec_raw else tuple()
    )
    nuts_path = args.nuts_gpkg
    if nuts_path is None:
        found = _first_existing(root, NUTS_CANDIDATES)
        if found is None:
            raise SystemExit(
                "NUTS geopackage not found. Pass --nuts-gpkg or place "
                "Data/geometry/NUTS_RG_20M_2021_3035.gpkg under the project root."
            )
        nuts_path = found
    else:
        nuts_path = nuts_path if nuts_path.is_absolute() else root / nuts_path
        if not nuts_path.is_file():
            raise SystemExit(f"NUTS gpkg not found: {nuts_path}")

    mainland_wgs = load_greece_mainland_wgs84(nuts_path, exclude)
    geom = mainland_wgs.geometry.iloc[0]
    b = mainland_wgs.total_bounds
    left, bottom, right, top = _expand_bounds(
        float(b[0]), float(b[1]), float(b[2]), float(b[3]), args.pad_deg
    )
    width, height = int(args.grid_width), int(args.grid_height)
    dst_t = from_bounds(left, bottom, right, top, width, height)
    mask_arr = _rasterize_mask(geom, dst_t, height, width)

    corine_file: Path | None = None
    if args.corine is not None:
        cp = args.corine if args.corine.is_absolute() else root / args.corine
        if cp.is_file():
            corine_file = cp
    if corine_file is None:
        corine_file = _first_existing(root, CORINE_CANDIDATES)

    landscan_path = args.landscan if args.landscan.is_absolute() else root / args.landscan

    fmap = folium.Map(
        location=[(bottom + top) / 2, (left + right) / 2],
        zoom_start=7,
        tiles=None,
    )
    folium.TileLayer(
        "CartoDB positron",
        name="Light (CartoDB Positron)",
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr=(
            "Tiles &copy; Esri &mdash; "
            "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
        ),
        name="Satellite (Esri World Imagery)",
        max_zoom=19,
        control=True,
    ).add_to(fmap)

    if not args.no_landscan:
        if not landscan_path.is_file():
            print(f"Skipping LandScan (file not found): {landscan_path}", file=sys.stderr)
        else:
            pop = _reproject_band_to_wgs84_grid(
                landscan_path,
                dst_t,
                (height, width),
                resampling="bilinear",
            )
            pop_rgba = population_to_rgba(pop, mask_arr)
            fg_ls = folium.FeatureGroup(name="LandScan 2020 population (grid)", show=True)
            folium.raster_layers.ImageOverlay(
                image=pop_rgba,
                bounds=[[bottom, left], [top, right]],
                mercator_project=True,
                opacity=1.0,
                name="LandScan",
                interactive=False,
                cross_origin=False,
            ).add_to(fg_ls)
            fg_ls.add_to(fmap)

    if not args.no_corine:
        corine_path = corine_file
        if corine_path is None or not corine_path.is_file():
            print(
                "Skipping CORINE (set --corine to your CLC GeoTIFF path).",
                file=sys.stderr,
            )
        else:
            corine_codes = _parse_corine_class_codes(args.corine_class_codes)
            legend_rgb = _parse_rgb_triplet(args.corine_legend_rgb)
            rgb_tol = int(args.corine_rgb_tolerance)
            cmode = args.corine_mode

            with rasterio.open(corine_path) as _cmeta:
                nband = int(_cmeta.count)
                if not args.quiet_corine_diagnostics:
                    dt = _cmeta.dtypes
                    dt_s = ", ".join(str(x) for x in dt[: min(6, len(dt))])
                    if len(dt) > 6:
                        dt_s += ", ..."
                    print(
                        f"CORINE file meta: bands={nband} dtype=[{dt_s}] "
                        f"nodata={_cmeta.nodata} crs={_cmeta.crs}",
                        file=sys.stderr,
                    )
                    print(
                        "  Nomenclature: CLC L2 code 121 = Industrial or commercial units and public facilities. "
                        "Values like 204,77,242,255 in a legend are usually RGBA symbology for 121, not extra codes.",
                        file=sys.stderr,
                    )

            if cmode == "rgb" and nband < 3:
                raise SystemExit(
                    f"{corine_path.name}: --corine-mode rgb needs >= 3 bands (file has {nband})."
                )

            u_all = np.zeros((height, width), dtype=bool)

            if cmode in ("class", "both"):
                clc = _reproject_band_to_wgs84_grid(
                    corine_path,
                    dst_t,
                    (height, width),
                    resampling="nearest",
                    band=int(args.corine_band),
                )
                if not args.quiet_corine_diagnostics:
                    log_corine_mainland_class_histogram(
                        clc,
                        mask_arr,
                        corine_path=corine_path,
                        class_codes=corine_codes,
                    )
                u_all |= industry_corine_mask_from_class(
                    clc, mask_arr, class_codes=corine_codes
                )

            if cmode in ("rgb", "both") and nband >= 3:
                rgb = _reproject_three_bands_to_wgs84_grid(
                    corine_path,
                    dst_t,
                    (height, width),
                    bands=(1, 2, 3),
                    resampling="nearest",
                )
                u_rgb = industry_corine_mask_from_legend_rgb(
                    rgb,
                    mask_arr,
                    target_rgb=legend_rgb,
                    tolerance=rgb_tol,
                )
                if not args.quiet_corine_diagnostics:
                    print(
                        f"  Legend RGB match {legend_rgb} +/- {rgb_tol}: "
                        f"{int(np.count_nonzero(u_rgb))} mainland px",
                        file=sys.stderr,
                    )
                u_all |= u_rgb
            elif cmode == "both" and nband < 3 and not args.quiet_corine_diagnostics:
                print(
                    "  Legend RGB match skipped (fewer than 3 bands; class codes only).",
                    file=sys.stderr,
                )

            if not args.quiet_corine_diagnostics:
                print(
                    f"  Combined industry pixels on map: {int(np.count_nonzero(u_all))}",
                    file=sys.stderr,
                )

            u_rgba = industry_mask_to_rgba(u_all)
            if cmode == "class":
                layer_bits = f"class {','.join(str(c) for c in corine_codes)}"
            elif cmode == "rgb":
                layer_bits = f"legend RGB {legend_rgb}"
            else:
                layer_bits = f"class {','.join(str(c) for c in corine_codes)}"
                if nband >= 3:
                    layer_bits += f" + RGB{legend_rgb}"
            fg_c = folium.FeatureGroup(
                name=f"CORINE industry ({layer_bits})",
                show=True,
            )
            folium.raster_layers.ImageOverlay(
                image=u_rgba,
                bounds=[[bottom, left], [top, right]],
                mercator_project=True,
                opacity=1.0,
                interactive=False,
                cross_origin=False,
            ).add_to(fg_c)
            fg_c.add_to(fmap)

    if vec_clcs:
        if corine_file is None or not corine_file.is_file():
            print(
                "Skipping CORINE vectors (GeoTIFF missing; set --corine or place file under CORINE_CANDIDATES).",
                file=sys.stderr,
            )
        else:
            lg = args.corine_vector_legend
            if lg is not None:
                legend_p = lg if lg.is_absolute() else root / lg
            else:
                legend_p = _first_existing(root, CORINE_LEGEND_CSV_CANDIDATES)
            clc_to_grid: dict[int, int] = {}
            clc_to_rgb: dict[int, tuple[int, int, int]] = {}
            clc_to_label: dict[int, str] = {}
            if legend_p is not None and legend_p.is_file():
                clc_to_grid, clc_to_rgb, clc_to_label = load_clc_legend_table(legend_p)
            else:
                print(
                    "CORINE vectors: CLC_legend.csv not found; using your codes as raw pixel values only "
                    "(no GRID_CODE alias).",
                    file=sys.stderr,
                )
            try:
                gdf_vec = polygonize_corine_classes_in_extent(
                    corine_file,
                    mainland_wgs,
                    (left, bottom, right, top),
                    band=int(args.corine_band),
                    selected_clcs=vec_clcs,
                    clc_to_grid=clc_to_grid,
                    simplify_m=float(args.corine_vector_simplify_m),
                )
            except Exception as exc:
                print(f"CORINE vector polygonize failed: {exc}", file=sys.stderr)
                gdf_vec = None
            if gdf_vec is not None and not gdf_vec.empty:
                if not args.quiet_corine_diagnostics:
                    print(
                        f"CORINE vectors: {len(gdf_vec)} features, classes {sorted(gdf_vec['clc'].unique().tolist())}",
                        file=sys.stderr,
                    )
                for clc in vec_clcs:
                    sub = gdf_vec[gdf_vec["clc"] == int(clc)]
                    if sub.empty:
                        continue
                    rgb = clc_to_rgb.get(int(clc), (140, 140, 140))
                    lab = clc_to_label.get(int(clc), "")
                    short = (lab[:42] + "...") if len(lab) > 43 else lab
                    layer_title = f"CORINE {clc}"
                    if short:
                        layer_title += f" {short}"
                    fg_v = folium.FeatureGroup(name=layer_title, show=True)
                    folium.GeoJson(
                        sub,
                        style_function=lambda _f, rr=rgb[0], gg=rgb[1], bb=rgb[2]: {
                            "fillColor": f"#{rr:02x}{gg:02x}{bb:02x}",
                            "color": "#222222",
                            "weight": 1,
                            "fillOpacity": 0.42,
                        },
                    ).add_to(fg_v)
                    fg_v.add_to(fmap)

    fg_outline = folium.FeatureGroup(name="Mainland outline (NUTS2, excl. islands)", show=True)
    folium.GeoJson(
        mainland_wgs,
        style_function=lambda _: {
            "fillColor": "#000000",
            "color": "#1a1a1a",
            "weight": 2,
            "fillOpacity": 0.0,
        },
    ).add_to(fg_outline)
    fg_outline.add_to(fmap)

    import importlib.util

    _cams_path = Path(__file__).resolve().parent / "cams_A_publicpower_greece.py"
    _spec = importlib.util.spec_from_file_location("cams_A_publicpower_greece", _cams_path)
    if _spec is None or _spec.loader is None:
        raise SystemExit(f"Cannot load CAMS helper: {_cams_path}")
    cams = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cams)
    DEFAULT_JRC_UNITS_CSV = cams.DEFAULT_JRC_UNITS_CSV
    DEFAULT_NC = cams.DEFAULT_NC
    IDX_A_PUBLIC_POWER = cams.IDX_A_PUBLIC_POWER
    _build_domain_mask = cams._build_domain_mask
    _country_index_1based = cams._country_index_1based
    add_cams_a_publicpower_layers_to_map = cams.add_cams_a_publicpower_layers_to_map
    load_jrc_open_units = cams.load_jrc_open_units

    nc_path = Path(DEFAULT_NC)
    if not nc_path.is_file():
        nc_path = root / DEFAULT_NC
    if not nc_path.is_file():
        raise SystemExit(f"CAMS NetCDF not found: {nc_path}")

    ds = xr.open_dataset(nc_path)
    try:
        grc_1b = _country_index_1based(ds, "GRC")
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        base = (emis == IDX_A_PUBLIC_POWER) & _build_domain_mask(
            lon, lat, ci, grc_1b, None
        )
        jrc_df = None
        jrc_suffix = ""
        if not args.no_jrc:
            jrc_p = (
                DEFAULT_JRC_UNITS_CSV
                if DEFAULT_JRC_UNITS_CSV.is_file()
                else root / DEFAULT_JRC_UNITS_CSV
            )
            if jrc_p.is_file():
                jrc_df = load_jrc_open_units(
                    jrc_p,
                    country_name="Greece",
                    bbox=None,
                    exclude_hydro_types=True,
                )
                jrc_suffix = " - excl. Hydro"
            else:
                print(f"Skipping JRC (CSV not found): {jrc_p}", file=sys.stderr)

        pol = args.pollutant.strip().lower()
        add_cams_a_publicpower_layers_to_map(
            fmap,
            ds,
            base,
            pol,
            jrc_df=jrc_df,
            jrc_layer_suffix=jrc_suffix,
            map_max_area_cells=max(0, int(args.map_max_area_cells)),
            map_sample_seed=42,
        )
    finally:
        ds.close()

    folium.LayerControl(collapsed=False).add_to(fmap)
    out = args.out_html if args.out_html.is_absolute() else root / args.out_html
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
