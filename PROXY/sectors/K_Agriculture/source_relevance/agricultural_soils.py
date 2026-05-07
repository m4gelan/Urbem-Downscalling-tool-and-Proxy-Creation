"""
Crop production NMVOC (NFR 3.D / GNFR L): EMEP Table 3.3 EFs per LUCAS LC1 with
Thunen NIR assignment rules.

Agricultural CO2 from urea and liming is handled in urea_application.py and
soil_liming.py.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from PROXY.sectors.K_Agriculture.tabular.emission_factors import load_nmvoc_crop_ef
from PROXY.sectors.K_Agriculture.k_config import project_root

from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points
from .lucas_survey import norm_lucas_str

# Phase 1.5: values sourced from PROXY/config/agriculture/emission_factors.yaml (nmvoc_crop_ef).
# (kg NMVOC ha-1 yr-1, EMEP/EEA Guidebook Table 3.3, Thunen NIR assignment).
NMVOC_EF: dict[str, float] = load_nmvoc_crop_ef()


def _perc_fraction(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return 1.0
    v = row[col]
    if pd.isna(v) or v == "":
        return 1.0
    try:
        return float(v) / 100.0
    except (TypeError, ValueError):
        return 1.0


def nmvoc_score_per_point(row: pd.Series) -> float:
    lc1 = norm_lucas_str(row.get("SURVEY_LC1"))
    lc2 = norm_lucas_str(row.get("SURVEY_LC2"))
    if lc1 in NMVOC_EF:
        return float(NMVOC_EF[lc1]) * _perc_fraction(row, "SURVEY_LC1_PERC")
    if lc2 in NMVOC_EF:
        return float(NMVOC_EF[lc2]) * _perc_fraction(row, "SURVEY_LC2_PERC")
    return 0.0


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = pts.apply(nmvoc_score_per_point, axis=1)
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
