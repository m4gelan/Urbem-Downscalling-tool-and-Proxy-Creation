from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import (
    pixels_inside_cams_cells,
    read_raster_window_for_cams,
)


def load_population(
    population_filepath: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
) -> tuple[np.ndarray, np.ndarray, Any, Any, float | None]:
    """Population, CAMS-cell mask, transform, CRS, and source nodata (if any)."""
    raw, transform, raster_crs, nodata = read_raster_window_for_cams(
        population_filepath, band, cams_cells
    )

    arr = raw.astype(np.float32, copy=True)
    if nodata is not None:
        arr[arr == np.float32(nodata)] = 0.0
    # Negative sentinels (common in pop rasters) → 0
    np.maximum(arr, 0.0, out=arr)

    height, width = arr.shape
    inside = pixels_inside_cams_cells(height, width, transform, raster_crs, cams_cells)

    arr[~inside] = 0.0

    n_pixels = int(np.count_nonzero(arr))
    log.info(
        f"Population : {n_pixels} non-zero pixels "
    )
    return arr, inside, transform, raster_crs, nodata
