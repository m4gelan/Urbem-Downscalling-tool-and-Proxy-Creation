from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from PROXY.core.raster.align import (
    ref_profile_to_kwargs as _core_ref_profile_to_kwargs,
)
from PROXY.core.raster.align import (
    warp_raster_to_ref as _core_warp_raster_to_ref,
)


def raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": None if src.crs is None else src.crs.to_string(),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0] if src.count else None,
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
            "nodata": src.nodata,
        }


def read_band(path: Path, band: int = 1, masked: bool = True) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band, masked=masked)


def warp_band_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    resampling: Resampling,
    band: int = 1,
) -> np.ndarray:
    """Reproject one raster band onto the reference grid (``ref`` from e.g. ``reference_window_profile``)."""
    h, w = int(ref["height"]), int(ref["width"])
    dst = np.zeros((h, w), dtype=np.float32)
    with rasterio.open(src_path) as src:
        nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=resampling,
            src_nodata=nodata,
            dst_nodata=np.nan,
        )
    out = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.maximum(out, 0.0)


def ref_profile_to_kwargs(ref: dict[str, Any]) -> dict[str, Any]:
    return _core_ref_profile_to_kwargs(ref)


def warp_raster_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    src_nodata: float | None = None,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """
    Read one band from ``src_path`` and reproject onto ``ref`` grid (height, width, transform, crs).

    Preserves NaN for no-data in the output (unlike ``warp_band_to_ref`` which zero-fills for Hotmaps-style stacks).
    """
    return _core_warp_raster_to_ref(
        src_path,
        ref,
        band=band,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )


def assert_same_grid(reference_path: Path, candidate_path: Path) -> None:
    with rasterio.open(reference_path) as ref, rasterio.open(candidate_path) as cand:
        same_shape = (ref.width == cand.width) and (ref.height == cand.height)
        same_crs = ref.crs == cand.crs
        same_transform = tuple(ref.transform) == tuple(cand.transform)
    if not (same_shape and same_crs and same_transform):
        raise ValueError(
            f"Grid mismatch between {reference_path.name} and {candidate_path.name}"
        )

