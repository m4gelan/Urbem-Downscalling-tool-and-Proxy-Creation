"""
Urea application (CO2): LUCAS cropland vs grassland baseline scores, grass scaled by livestock
intensity at NUTS-2 -> mean mu per NUTS-2 x CLC -> rho (country max-normalization).

Cropland points: s_p = 1.0. Eligible grassland: s_p = 0.7 * omega_n, with omega from
I_grass = N_bovine + N_sheep + N_goats (C21 head counts) / median_positive(I); missing NUTS -> 1.0;
no positive median -> omega = 1.0 everywhere. No clipping of omega (unlike enteric/housing census omega).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.k_config import project_root

from .census_intensity import load_c21_counts_for_cfg
from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .fertilized_land import eligible_synthetic_n
from .lucas_points import get_lucas_ag_points
from .lucas_survey import norm_lucas_str

_GRASS_SPECIES = ("bovine", "sheep", "goats")


def grass_livestock_intensity(counts: pd.DataFrame) -> pd.Series:
    """I_c^grass = N_bovine + N_sheep + N_goats per NUTS-2 row (C21 thousands as stored)."""
    tot = pd.Series(0.0, index=counts.index, dtype=float)
    for sp in _GRASS_SPECIES:
        if sp not in counts.columns:
            continue
        tot = tot + pd.to_numeric(counts[sp], errors="coerce").fillna(0.0).astype(float)
    return tot


def grass_livestock_omega(intensity: pd.Series) -> pd.Series:
    """
    omega_c^grass = I_c / median_positive(I). Cells with I <= 0 keep ratio 0 (not clipped).
    If no strictly positive values or median is non-finite, return 1.0 for all indices.
    """
    pos = intensity[intensity > 0]
    med = float(pos.median()) if len(pos) else float("nan")
    if not np.isfinite(med) or med <= 0:
        return pd.Series(1.0, index=intensity.index, dtype=float)
    w = intensity.astype(float) / med
    return w


def load_grass_livestock_omega_for_cfg(cfg: dict[str, Any], root: Path) -> pd.Series:
    """
    Build omega_n per NUTS-2 from C21 using paths in cfg['paths']['census'].
    NUTS not present in C21 layers are handled at map time (fallback 1.0).
    """
    counts = load_c21_counts_for_cfg(cfg, root)
    i_grass = grass_livestock_intensity(counts)
    omega = grass_livestock_omega(i_grass)
    omega.index = omega.index.astype(str).str.strip()
    omega.name = "omega_grass"
    return omega


def point_urea_score(
    lc1: Any,
    lu1: Any,
    nuts_id: Any,
    omega_by_nuts: pd.Series,
    *,
    missing_omega_fallback: float = 1.0,
) -> float:
    """
    Point-level s_p^urea: cropland (same LC1/LU1 eligibility as synthetic N) -> 1.0;
    eligible grassland -> 0.7 * omega(NUTS_ID); else excluded (NaN).
    """
    if not eligible_synthetic_n(lc1, lu1):
        return float("nan")
    lc = norm_lucas_str(lc1)
    if lc.startswith("B"):
        return 1.0
    nid = str(nuts_id).strip() if nuts_id is not None and not (isinstance(nuts_id, float) and np.isnan(nuts_id)) else ""
    if not nid:
        return float("nan")
    idx = pd.Index(omega_by_nuts.index.astype(str).str.strip())
    omega_s = pd.Series(omega_by_nuts.values, index=idx)
    w = omega_s.get(nid, np.nan)
    if not np.isfinite(w):
        w = missing_omega_fallback
    return 0.7 * float(w)


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    # Baseline: eligible cropland (LC1 B*) -> 1.0; eligible grassland -> 0.7 * omega_n (NUTS-2 from C21).
    # mu_{n,c}: mean s_p over retained LUCAS points per (NUTS_ID, CLC_CODE); rho via merge_extent + country max-norm.
    root = project_root(cfg)
    omega = load_grass_livestock_omega_for_cfg(cfg, root)
    idx = pd.Index(omega.index.astype(str).str.strip())
    omega_n = pd.Series(omega.values, index=idx)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    nuts_k = pts["NUTS_ID"].astype(str).str.strip()
    w_cell = pd.to_numeric(nuts_k.map(omega_n), errors="coerce").fillna(1.0)
    scores: list[float] = []
    for lc, lu, w in zip(pts["SURVEY_LC1"], pts["SURVEY_LU1"], w_cell):
        if not eligible_synthetic_n(lc, lu):
            scores.append(float("nan"))
            continue
        lc_n = norm_lucas_str(lc)
        if lc_n.startswith("B"):
            scores.append(1.0)
        else:
            scores.append(0.7 * float(w))
    pts["mu"] = scores
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
