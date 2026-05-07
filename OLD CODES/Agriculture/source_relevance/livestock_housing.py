"""
Livestock housing (NH3, GNFR K): LUCAS LU1/LC1 stage-1 score, census omega stage-2 -> mu per NUTS-2 x CLC -> rho.

LUCAS 2022: U111 = agriculture; U112 = fallow; U113 = kitchen garden (excluded); U120 = forestry (excluded).

Emission factors and C21 column names load from JSON only (see config/emission_factors/).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from Agriculture.config import project_root

from .census_intensity import load_census_omega_for_cfg
from .common import aggregate_nuts_clc_mu, apply_census_omega_to_agg, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points
from .lucas_survey import norm_lucas_str

MISSING_G = 0.4

_AGRI_LU_FALLBACK = frozenset({"U111", "U112"})
_BUILDING_LC = frozenset({"A11", "A12"})
_GRASSLAND_LC_PREFIX = "E"
_CROPLAND_LC_PREFIX = "B"
_LEY_LC = "B55"


def _parse_grazing_code(row: Union[pd.Series, dict[str, Any]]) -> int | None:
    g_raw = row.get("SURVEY_GRAZING", np.nan)  # type: ignore[union-attr]
    if g_raw is None or (isinstance(g_raw, float) and np.isnan(g_raw)):
        return None
    try:
        g = int(float(g_raw))
    except (ValueError, TypeError):
        return None
    if g not in (0, 1, 2):
        return None
    return g


def point_livestock_housing_nh3(row: Union[pd.Series, dict[str, Any]]) -> Optional[float]:
    """
    Stage 1: NH3 housing proxy s_p in [0,1] from LUCAS (no emission factors here).

    g=1 (grazing observed) -> 0.0. Other tiers use confirmed scores for g in {0,2}
    and a damped score when g is missing.
    """
    lc = norm_lucas_str(row.get("SURVEY_LC1", ""))  # type: ignore[union-attr]
    lu = norm_lucas_str(row.get("SURVEY_LU1", ""))  # type: ignore[union-attr]

    if lu in ("U113", "U120"):
        return None

    g = _parse_grazing_code(row)

    if g == 1:
        return 0.0

    confirmed = g is not None and g in (0, 2)

    if lc in _BUILDING_LC and lu == "U111":
        tier = 1.0
    elif lu == "U111" and lc.startswith(_GRASSLAND_LC_PREFIX):
        tier = 0.55
    elif lu == "U111" and lc.startswith(_CROPLAND_LC_PREFIX) and lc != _LEY_LC:
        tier = 0.20
    elif lu == "U111" and lc == _LEY_LC:
        tier = 0.30
    elif lu == "U112":
        tier = 0.20
    else:
        return None

    sp = tier if confirmed else MISSING_G * tier
    return None if (isinstance(sp, float) and math.isnan(sp)) else float(sp)


def _mu_scalar(x: Optional[float]) -> float:
    if x is None:
        return float("nan")
    return float(x)


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = pts.apply(point_livestock_housing_nh3, axis=1)
    agg = aggregate_nuts_clc_mu(pts, "mu")
    omega, fb = load_census_omega_for_cfg(cfg, root, "housing")
    agg = apply_census_omega_to_agg(agg, omega, fb)
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
