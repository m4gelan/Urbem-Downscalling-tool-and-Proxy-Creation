"""
Enteric fermentation / grazing: LUCAS grazing survey -> point metric -> mean mu per NUTS-2 x CLC -> rho.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.k_config import project_root

from .census_intensity import load_census_omega_for_cfg
from .common import aggregate_nuts_clc_mu, apply_census_omega_to_agg, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points


def _norm_lc1(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().strip('"').strip("'").upper()
    t = " ".join(t.split())
    if t in ("NAN", "NONE", "NA", "#N/A", "NAT"):
        return ""
    return t


def point_grazing_metric(graze: Any, lc1: Any) -> float:
    """Grazing intensity proxy for one LUCAS point (used by enteric CH4 and by manure module)."""
    if pd.isna(graze):
        lc = _norm_lc1(lc1)
        if lc.startswith("E") or lc == "B55":
            return 0.6
        return float("nan")
    g = float(graze)
    if g == 1.0:
        return 1.0
    if g in (0.0, 2.0):
        return 0.0
    return float("nan")


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = [
        point_grazing_metric(g, lc) for g, lc in zip(pts["SURVEY_GRAZING"], pts["SURVEY_LC1"])
    ]
    agg = aggregate_nuts_clc_mu(pts, "mu")
    omega, fb = load_census_omega_for_cfg(cfg, root, "enteric")
    agg = apply_census_omega_to_agg(agg, omega, fb)
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
