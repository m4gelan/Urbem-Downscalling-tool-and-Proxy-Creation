"""Country- and pollutant-specific composite proxy P = w_solid*P_solid + w_ww*P_ww + w_res*P_res."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from . import io_utils

logger = logging.getLogger(__name__)


def apply_point_source_mask(
    P: np.ndarray,
    mask: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray:
    """
    Optional fine-grid mask (0–1). ``mode: multiply`` scales proxy;
    ``alpha`` scales mask strength (1 = full effect).
    """
    if mask is None:
        return P
    mcfg = (cfg.get("proxy") or {}).get("point_source_mask") or {}
    mode = str(mcfg.get("mode", "multiply")).lower()
    alpha = float(mcfg.get("alpha", 1.0))
    m = np.clip(np.asarray(mask, dtype=np.float64), 0.0, 1.0)
    if mode == "multiply":
        return (P * (1.0 - alpha * (1.0 - m))).astype(np.float32)
    return (P * m).astype(np.float32)


def load_mask_optional(cfg: dict[str, Any], ref: dict[str, Any]) -> np.ndarray | None:
    root = cfg["_project_root"]
    p = (cfg.get("paths") or {}).get("point_source_mask_tif")
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        logger.info("point_source_mask_tif not found (%s); skipping.", path)
        return None
    return io_utils.warp_raster_to_ref(path, ref, resampling=Resampling.bilinear).astype(np.float32)


def composite_per_pollutant(
    proxy_solid: np.ndarray,
    proxy_ww: np.ndarray,
    proxy_res: np.ndarray,
    country_id: np.ndarray,
    ws: np.ndarray,
    ww: np.ndarray,
    wr: np.ndarray,
    pollutants: list[str],
    mask: np.ndarray | None,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    ``ws``, ``ww``, ``wr`` shaped ``(n_iso, n_pol)`` aligned with ``country_id`` values
    as row indices into ISO3 list (including index 0 for outside).
    """
    cid = np.clip(country_id.astype(np.int64), 0, ws.shape[0] - 1)
    out: dict[str, np.ndarray] = {}
    for j, pol in enumerate(pollutants):
        w_s = ws[cid, j]
        w_w = ww[cid, j]
        w_r = wr[cid, j]
        P = (
            w_s.astype(np.float32) * proxy_solid
            + w_w.astype(np.float32) * proxy_ww
            + w_r.astype(np.float32) * proxy_res
        )
        P = np.maximum(P, 0.0)
        P = apply_point_source_mask(P, mask, cfg)
        out[pol.lower()] = P.astype(np.float32)
    return out
