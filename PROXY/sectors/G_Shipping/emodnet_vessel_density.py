"""
EMODnet HA vessel density GeoTIFF: warp to a reference grid (EPSG:3035 EMODnet typical).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from PROXY.core.raster.align import ref_profile_to_kwargs


def read_vessel_density_metadata(tif_path: Path) -> dict[str, Any]:
    with rasterio.open(tif_path) as src:
        epsg = None
        if src.crs:
            try:
                epsg = src.crs.to_epsg()
            except Exception:
                epsg = None
        return {
            "path": str(tif_path.resolve()),
            "crs_wkt": src.crs.to_wkt() if src.crs else None,
            "epsg": epsg,
            "width": int(src.width),
            "height": int(src.height),
            "transform": [float(x) for x in src.transform[:6]],
            "bounds": {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            },
            "nodata": float(src.nodata)
            if src.nodata is not None and np.isfinite(src.nodata)
            else None,
            "dtype": src.dtypes[0],
            "count": int(src.count),
        }


def _sanitize_band(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        a = np.where(a == np.float32(nodata), np.nan, a)
    a = np.where(np.abs(a) > 1e38, np.nan, a)
    return a


def warp_vessel_density_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    kw = ref_profile_to_kwargs(ref)
    h, w = kw["height"], kw["width"]
    dst_transform = kw["transform"]
    dst_crs = kw["crs"]
    out = np.full((h, w), dst_nodata, dtype=np.float32)
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, dst_transform)
    with rasterio.open(src_path) as src:
        nodata = src.nodata
        west, south, east, north = transform_bounds(
            dst_crs, src.crs, left, bottom, right, top, densify_pts=21
        )
        win = from_bounds(west, south, east, north, transform=src.transform)
        try:
            win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        except WindowError:
            return out
        if win.width < 1 or win.height < 1:
            return out
        arr = src.read(band, window=win)
        arr = _sanitize_band(arr, nodata)
        src_transform = src.window_transform(win)
        reproject(
            source=arr,
            destination=out,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out
