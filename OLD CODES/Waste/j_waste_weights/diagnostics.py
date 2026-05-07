"""CSV and optional GeoTIFF side outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


def write_pollutant_band_mapping(path: Path, pollutants: list[str]) -> None:
    rows = [{"band_index": i + 1, "pollutant": p, "nc_name": p.lower()} for i, p in enumerate(pollutants)]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Wrote %s", path)


def write_country_pollutant_weights(path: Path, wide: pd.DataFrame) -> None:
    wide.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def write_fallback_log(path: Path, fb: pd.DataFrame | None) -> None:
    if fb is None or not len(fb):
        pd.DataFrame(columns=["country_iso3", "pollutant", "tier", "note"]).to_csv(path, index=False)
        logger.info("Wrote empty %s", path)
        return
    fb.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def write_zero_proxy_diagnostics(
    path: Path,
    cam_cell_id: np.ndarray,
    fallback_mask: np.ndarray,
) -> None:
    """One row per CAMS cell that used uniform fallback (any fine pixel)."""
    cid = cam_cell_id.ravel()
    fb = fallback_mask.ravel()
    m = fb & (cid >= 0)
    if not np.any(m):
        pd.DataFrame(columns=["cam_cell_id", "n_pixels_fallback"]).to_csv(path, index=False)
        logger.info("Wrote empty %s", path)
        return
    cells, counts = np.unique(cid[m], return_counts=True)
    pd.DataFrame({"cam_cell_id": cells, "n_pixels_fallback": counts}).to_csv(path, index=False)
    logger.info("Wrote %s (%d CAMS cells)", path, len(cells))


def write_geotiff_single(
    path: Path,
    arr: np.ndarray,
    ref: dict[str, Any],
    *,
    nodata: float | None = None,
) -> None:
    h, w = arr.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": arr.dtype,
        "crs": CRS.from_string(ref["crs"]),
        "transform": ref["transform"],
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def write_multiband_weights(
    path: Path,
    bands: dict[str, np.ndarray],
    pollutants_order: list[str],
    ref: dict[str, Any],
) -> None:
    pols = [p.lower() for p in pollutants_order]
    arrs = [bands[p] for p in pols]
    stack = np.stack(arrs, axis=0)
    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": "float32",
        "crs": CRS.from_string(ref["crs"]),
        "transform": ref["transform"],
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, pol in enumerate(pols, start=1):
            dst.write(stack[i - 1].astype(np.float32), i)
            dst.set_band_description(i, f"j_waste_weight_{pol}")
