"""CORINE-aligned reference grid window from NUTS country union."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.windows import Window, from_bounds


_CORINE_CANDIDATES = [
    Path("data/CORINE/U2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
    Path("data/CORINE/\u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
    Path("Input/CORINE/U2018_CLC2018_V2020_20u1.tif"),
]


def resolve_path(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)


def iter_corine_search_paths(root: Path, configured: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if configured:
        paths.append(resolve_path(root, configured))
    for rel in _CORINE_CANDIDATES:
        paths.append(root / rel)
    return paths


def first_existing_corine(root: Path, configured: str | Path | None) -> Path:
    for p in iter_corine_search_paths(root, configured):
        if p.is_file():
            return p
    raise FileNotFoundError(
        "CORINE GeoTIFF not found. Set paths.corine in config or place data under data/CORINE/."
    )


def nuts2_for_country(gpkg: Path, cntr: str) -> gpd.GeoDataFrame:
    nuts = gpd.read_file(gpkg)
    if nuts.crs is None:
        raise ValueError("NUTS GeoPackage has no CRS.")
    n2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    if "CNTR_CODE" not in n2.columns:
        raise ValueError("GeoPackage missing CNTR_CODE.")
    cc = n2["CNTR_CODE"].astype(str).str.strip().str.upper()
    n2 = n2[cc == str(cntr).strip().upper()].copy()
    if n2.empty:
        raise ValueError(f"No NUTS-2 rows for CNTR_CODE={cntr!r}.")
    return n2


def reference_window_profile(
    *,
    corine_path: Path,
    nuts_gpkg: Path,
    nuts_cntr: str,
    pad_m: float = 5000.0,
) -> dict[str, Any]:
    """
    Return dict: corine_path, transform, height, width, crs (str), window_bounds_3035,
    domain_bbox_wgs84 (west,south,east,north).
    """
    from rasterio.warp import transform_bounds

    n2 = nuts2_for_country(nuts_gpkg, nuts_cntr)
    union = n2.dissolve().geometry.iloc[0]
    if union is None or union.is_empty:
        raise ValueError("NUTS union is empty.")

    with rasterio.open(corine_path) as src:
        if src.crs is None:
            raise ValueError(f"CORINE has no CRS: {corine_path}")
        crs = src.crs
        g3035 = gpd.GeoDataFrame(geometry=[union], crs=n2.crs).to_crs(crs)
        geom = g3035.geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        pad = float(pad_m)
        win = from_bounds(
            minx - pad, miny - pad, maxx + pad, maxy + pad, transform=src.transform
        )
        win = win.round_lengths().intersection(
            Window(0, 0, src.width, src.height)
        )
        if win.width < 1 or win.height < 1:
            raise ValueError("CORINE window empty after NUTS clip.")
        transform = src.window_transform(win)
        height, width = int(win.height), int(win.width)
        left, bottom, right, top = rasterio.transform.array_bounds(
            height, width, transform
        )
        w, s, e, n = transform_bounds(
            crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
        )

    return {
        "corine_path": corine_path,
        "transform": transform,
        "height": height,
        "width": width,
        "crs": crs.to_string(),
        "window_bounds_3035": (left, bottom, right, top),
        "domain_bbox_wgs84": (w, s, e, n),
    }
