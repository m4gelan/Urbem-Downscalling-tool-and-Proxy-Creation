"""
Manure management: LUCAS grazing + LU1/LC1 land-application weights -> mu per NUTS-2 x CLC -> rho.

Land-application weights follow Velthof et al. (2009) crop groups and grassland vs arable
differentiation; residual agricultural land (not E* / B*) uses a low weight consistent with
limited manure reception (cf. Oenema et al. 2007). Weights are < 1 and are interpreted as
probabilities of likely manure application, not confirmed use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from PROXY.sectors.K_Agriculture.tabular.emission_factors import load_manure_land_application
from PROXY.sectors.K_Agriculture.k_config import project_root

from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .enteric_grazing import point_grazing_metric
from .lucas_points import get_lucas_ag_points

# Phase 1.5: weights loaded from PROXY/config/agriculture/emission_factors.yaml (manure_land_application).
# manure_land_application.json (JSON defaults match these Python values exactly).
_MANURE_W = load_manure_land_application()

# Velthof et al. (2009): high manure-use arable (LC1); B2* = all codes with prefix B2.
_HIGH_MANURE_LC1_EXACT = frozenset({"B11", "B13", "B32"})
# Intermediate manure-use group.
_INTERMEDIATE_MANURE_LC1 = frozenset(
    {"B12", "B14", "B15", "B16", "B17", "B18", "B19", "B31"}
)


def _norm(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().strip('"').strip("'").upper()
    t = " ".join(t.split())
    if t in ("NAN", "NONE", "NA", "#N/A", "NAT"):
        return ""
    return t


def _arable_land_application_weight(lc: str) -> float:
    """
    Per-hectare-style relative manure-application weight for arable cropland (LC1 in B*).
    Order: resolve intermediate codes first, then high (including B2*), then residual B*.
    """
    if lc in _INTERMEDIATE_MANURE_LC1:
        return _MANURE_W["intermediate_manure"]
    if lc in _HIGH_MANURE_LC1_EXACT:
        return _MANURE_W["high_manure_exact"]
    if lc.startswith("B2"):
        return _MANURE_W["high_manure_prefix"]
    if lc.startswith("B"):
        return _MANURE_W["residual_b_prefix"]
    return 0.0


def point_manure_metric(graze: Any, lc1: Any, lu1: Any) -> float:
    """
    Combine grazing survey signal with LU1/LC1 land-application relevance (Velthof 2009).

    Tiers (land-application component):
      - Managed grassland (LU1 U111, LC1 E*): 0.8
      - Arable (LU1 U111, LC1 B*): 0.8 / 0.6 / 0.1 by crop group
      - Fallow (LU1 U112): 0.7
      - Other agricultural (LU1 U111 or U113, LC1 not E* and not B*): 0.1
      - U113 with LC1 B*: same arable tiers as U111 (extension; not in thesis bullet list)
      - U113 with LC1 E*: 0.1 (kitchen-garden / small-holder context; rare)
    """
    g_raw = point_grazing_metric(graze, lc1)
    lu = _norm(lu1)
    lc = _norm(lc1)

    lu_m = 0.0
    is_e = lc.startswith("E")
    is_b = lc.startswith("B")

    if lu == "U112":
        lu_m = _MANURE_W["fallow"]
    elif lu == "U111" and is_e:
        lu_m = _MANURE_W["managed_grassland"]
    elif lu == "U111" and is_b:
        lu_m = _arable_land_application_weight(lc)
    elif lu == "U111":
        lu_m = _MANURE_W["u111_other"]
    elif lu == "U113" and is_b:
        lu_m = _arable_land_application_weight(lc)
    elif lu == "U113" and is_e:
        lu_m = _MANURE_W["u113_other"]
    elif lu == "U113":
        lu_m = _MANURE_W["u113_other"]

    if not np.isnan(g_raw):
        return max(float(g_raw), lu_m)
    return lu_m if lu_m > 0 else float("nan")


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = [
        point_manure_metric(g, lc, lu)
        for g, lc, lu in zip(pts["SURVEY_GRAZING"], pts["SURVEY_LC1"], pts["SURVEY_LU1"])
    ]
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
