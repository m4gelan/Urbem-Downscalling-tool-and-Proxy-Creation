from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells, read_raster_window_for_cams


def resolve_imperviousness_filepath(rel_path: str | Path, country_profile: dict[str, str]) -> str:
    """Replace ``ISO3`` in the config path with ``country_profile['ISO3']``."""
    iso3 = str(country_profile["ISO3"]).strip().upper()
    if not iso3:
        raise ValueError("country_profile['ISO3'] is empty")
    p = str(rel_path).replace("\\", "/")
    if "ISO3" not in p:
        raise ValueError(f"Imperviousness path must contain ISO3 placeholder: {p!r}")
    return p.replace("ISO3", iso3)


def load_imperviousness(
    imperviousness_filepath: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    band: int = 1,
) -> tuple[np.ndarray, Any, Any, float | None]:
    """
    Imperviousness GeoTIFF clipped to the CAMS country footprint (same window as population).

    Logs finite-data **min** and **max** (ignores nodata when provided).
    """
    raw, transform, raster_crs, nodata = read_raster_window_for_cams(
        imperviousness_filepath, band, cams_cells
    )
    arr = np.asarray(raw, dtype=np.float64)
    if nodata is not None:
        arr = np.where(arr == float(nodata), np.nan, arr)

    h, w = arr.shape
    inside = pixels_inside_cams_cells(h, w, transform, raster_crs, cams_cells)
    m = inside & np.isfinite(arr)
    if np.any(m):
        vmin = float(np.nanmin(arr[m]))
        vmax = float(np.nanmax(arr[m]))
        log.info(f"Imperviousness raster (band {band}): min={vmin:.6g} max={vmax:.6g} (CAMS-window, finite pixels)")
    else:
        log.warning("Imperviousness raster: no finite pixels inside CAMS footprint in window")

    arr_out = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    arr_out[~inside] = 0.0
    return arr_out, transform, raster_crs, nodata


def load_ghsl_smod(
    ghsl_filepath: Path,
    cams_cells: dict[int, dict[str, Any]],
    *,
    rural_codes: list[int],
    band: int = 1,
) -> tuple[np.ndarray, Any, Any]:
    """
    GHSL SMOD class raster (windowed like population). Output is **float32 0/1**: ``1`` where the
    raw class value is in *rural_codes*, else ``0`` (masked to CAMS footprint).
    """
    codes = {int(c) for c in rural_codes}
    raw, transform, raster_crs, nodata = read_raster_window_for_cams(ghsl_filepath, band, cams_cells)
    cls = np.asarray(raw, dtype=np.int64)
    if nodata is not None:
        cls = np.where(cls == int(nodata), -9999, cls)

    out = np.isin(cls, list(codes)).astype(np.float32)
    h, w = out.shape
    inside = pixels_inside_cams_cells(h, w, transform, raster_crs, cams_cells)
    out = np.where(inside, out, 0.0)
    n_one = int(np.sum(out > 0.5))
    log.info(
        f"GHSL SMOD rural mask (codes {sorted(codes)}): {n_one} pixels == 1 on {h}x{w} CAMS window"
    )
    return out, transform, raster_crs
