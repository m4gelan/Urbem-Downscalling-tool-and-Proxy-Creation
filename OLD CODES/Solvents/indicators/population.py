from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from ..io.paths import resolve_path as project_resolve


def warp_population_to_ref(
    root: Path,
    pop_path: str | Path,
    ref: dict,
) -> np.ndarray:
    """Resample population raster to ref grid; non-finite -> 0; float32."""
    p = project_resolve(root, Path(pop_path))
    h, w = int(ref["height"]), int(ref["width"])
    dst = np.zeros((h, w), dtype=np.float32)
    if not p.is_file():
        return dst
    with rasterio.open(p) as src:
        nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=Resampling.sum,
            src_nodata=nodata,
            dst_nodata=np.nan,
        )
    out = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.maximum(out, 0.0)
