"""CSV and optional GeoTIFF side outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.core.io import write_geotiff

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
    write_geotiff(
        path=path,
        array=np.asarray(arr),
        crs=str(ref["crs"]),
        transform=ref["transform"],
        nodata=nodata,
        tiled=True,
        bigtiff="IF_SAFER",
    )
    logger.info("Wrote %s", path)


def write_multiband_weights(
    path: Path,
    bands: dict[str, np.ndarray],
    pollutants_order: list[str],
    ref: dict[str, Any],
) -> None:
    pols = [p.lower() for p in pollutants_order]
    arrs = [bands[p] for p in pols]
    stack = np.stack(arrs, axis=0)
    write_geotiff(
        path=path,
        array=stack.astype(np.float32, copy=False),
        crs=str(ref["crs"]),
        transform=ref["transform"],
        band_descriptions=[f"j_waste_weight_{pol}" for pol in pols],
        tiled=True,
        bigtiff="IF_SAFER",
    )
    logger.info("Wrote %s (%d bands)", path, len(pols))
