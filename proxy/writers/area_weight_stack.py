from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def _gtiff_meta(h: int, w: int, count: int, transform: Any, crs: Any) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }


def open_area_weight_stack(
    out_path: Path,
    height: int,
    width: int,
    band_count: int,
    transform: Any,
    crs: Any,
) -> rasterio.DatasetWriter:
    """Open a multi-band GeoTIFF; write planes with ``write_area_weight_plane`` then close."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return rasterio.open(out_path, "w", **_gtiff_meta(height, width, band_count, transform, crs))


def write_area_weight_plane(dst: rasterio.DatasetWriter, band_index: int, name: str, plane: np.ndarray) -> None:
    dst.write(np.asarray(plane, dtype=np.float32), int(band_index))
    dst.set_band_description(int(band_index), str(name))


def write_area_weight_equal_multiband(
    out_path: Path,
    values_2d: np.ndarray,
    band_names: list[str],
    transform: Any,
    crs: Any,
) -> Path:
    """GeoTIFF with one band per pollutant name; each band duplicates ``values_2d`` (float32)."""
    if not band_names:
        raise ValueError("band_names must be non-empty")
    h, w = values_2d.shape
    count = len(band_names)
    v = np.asarray(values_2d, dtype=np.float32)
    stack = np.broadcast_to(v, (count, h, w)).copy()
    with rasterio.open(out_path, "w", **_gtiff_meta(h, w, count, transform, crs)) as dst:
        dst.write(stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, str(name))
    return out_path


def write_area_weight_stack_multiband(
    out_path: Path,
    bands: np.ndarray,
    band_names: list[str],
    transform: Any,
    crs: Any,
) -> Path:
    """GeoTIFF from ``(count, height, width)`` stack — allocates full 3-D array."""
    stack = np.asarray(bands, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"bands must be 3-D (count, h, w), got shape {stack.shape}")
    count, h, w = stack.shape
    if len(band_names) != count:
        raise ValueError(f"band_names length {len(band_names)} != bands.shape[0] {count}")
    with rasterio.open(out_path, "w", **_gtiff_meta(h, w, count, transform, crs)) as dst:
        dst.write(stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, str(name))
    return out_path
