"""Country- and pollutant-specific composite proxy P = w_solid*P_solid + w_ww*P_ww + w_res*P_res."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from PROXY.core.raster import warp_raster_to_ref

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
    return warp_raster_to_ref(path, ref, resampling=Resampling.bilinear).astype(np.float32)


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
    ``ws``, ``ww``, ``wr`` shaped ``(n_iso, n_pol)`` (national / reported family shares).

    At each fine pixel, only families with **positive** local proxy receive their CEIP share;
    shares are **renormalized** to sum to 1 over those active families, then
    P = w'_s P_solid + w'_w P_ww + w'_r P_res. This matches reported alphas to local signal
    without zeroing the composite when one dominant family is absent in a cell.
    """
    ps = np.nan_to_num(np.asarray(proxy_solid, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pww = np.nan_to_num(np.asarray(proxy_ww, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pr = np.nan_to_num(np.asarray(proxy_res, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ps = np.maximum(ps, 0.0)
    pww = np.maximum(pww, 0.0)
    pr = np.maximum(pr, 0.0)
    thr = 1e-20
    m_s = (ps > thr).astype(np.float64)
    m_w = (pww > thr).astype(np.float64)
    m_r = (pr > thr).astype(np.float64)
    cid = np.clip(country_id.astype(np.int64), 0, ws.shape[0] - 1)
    out: dict[str, np.ndarray] = {}
    for j, pol in enumerate(pollutants):
        w_s = np.asarray(ws[cid, j], dtype=np.float64)
        w_w = np.asarray(ww[cid, j], dtype=np.float64)
        w_r = np.asarray(wr[cid, j], dtype=np.float64)
        wa = w_s * m_s
        wb = w_w * m_w
        wc = w_r * m_r
        sumw = wa + wb + wc
        num = wa * ps + wb * pww + wc * pr
        P = np.zeros_like(num, dtype=np.float64)
        pos = sumw > 0.0
        P[pos] = num[pos] / sumw[pos]
        n_act = m_s + m_w + m_r
        need_mean = (~pos) & (n_act > 0.0)
        P[need_mean] = (m_s * ps + m_w * pww + m_r * pr)[need_mean] / n_act[need_mean]
        P = P.astype(np.float32)
        P = apply_point_source_mask(P, mask, cfg)
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        out[pol.lower()] = P.astype(np.float32)
    return out
