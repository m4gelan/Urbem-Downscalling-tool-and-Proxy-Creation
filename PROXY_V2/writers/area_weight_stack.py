from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def write_area_weight_equal_multiband(
    out_path: Path,
    values_2d: np.ndarray,
    band_names: list[str],
    transform: Any,
    crs: Any,
) -> Path:
    """
    GeoTIFF with one band per pollutant name; each band duplicates ``values_2d`` (float32).
    """
    if not band_names:
        raise ValueError("band_names must be non-empty")
    h, w = values_2d.shape
    count = len(band_names)
    v = np.asarray(values_2d, dtype=np.float32)
    stack = np.broadcast_to(v, (count, h, w)).copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
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
    with rasterio.open(out_path, "w", **meta) as dst:
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
    """
    GeoTIFF with one band per row of ``bands`` — shape ``(count, height, width)``, float32.
    """
    stack = np.asarray(bands, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"bands must be 3-D (count, h, w), got shape {stack.shape}")
    count, h, w = stack.shape
    if len(band_names) != count:
        raise ValueError(f"band_names length {len(band_names)} != bands.shape[0] {count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
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
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, str(name))
    return out_path
