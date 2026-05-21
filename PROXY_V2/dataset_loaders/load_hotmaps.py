from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PROXY_V2.core import log
from PROXY_V2.dataset_loaders.load_cams_cells_mask import (
    pixels_inside_cams_cells,
    read_raster_window_for_cams,
)


def _load_hotmaps_band(
    path: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
    label: str,
) -> tuple[np.ndarray, Any, Any, float | None]:
    raw, transform, raster_crs, nodata = read_raster_window_for_cams(path, band, cams_cells)
    arr = np.asarray(raw, dtype=np.float64)
    if nodata is not None:
        arr = np.where(arr == float(nodata), np.nan, arr)
    np.maximum(arr, 0.0, out=arr)
    h, w = arr.shape
    inside = pixels_inside_cams_cells(h, w, transform, raster_crs, cams_cells)
    arr[~inside] = np.nan
    finite = inside & np.isfinite(arr)
    if np.any(finite):
        log.info(
            f"Hotmaps {label}: min={float(np.nanmin(arr[finite])):.6g} "
            f"max={float(np.nanmax(arr[finite])):.6g} ({h}x{w})"
        )
    else:
        log.warning(f"Hotmaps {label}: no finite pixels in CAMS window")
    return arr.astype(np.float32), transform, raster_crs, nodata


def load_hotmaps_heat_res(
    path: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
) -> tuple[np.ndarray, Any, Any, float | None]:
    return _load_hotmaps_band(path, cams_cells, band=band, label="H_res")


def load_hotmaps_heat_nres(
    path: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
) -> tuple[np.ndarray, Any, Any, float | None]:
    return _load_hotmaps_band(path, cams_cells, band=band, label="H_nres")


def load_hotmaps_hdd(
    path: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
) -> tuple[np.ndarray, Any, Any, float | None]:
    return _load_hotmaps_band(path, cams_cells, band=band, label="HDD")
