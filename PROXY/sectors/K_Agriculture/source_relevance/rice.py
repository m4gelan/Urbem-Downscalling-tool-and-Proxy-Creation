"""
Rice paddies (CH4, GNFR L): LUCAS LC1 B17 (rice) -> mu per NUTS-2 x CLC -> rho.

Point score is land-cover identity only (1.0 for B17); water-management EF variation
is not inferred from LUCAS and is left to the national inventory EF layer.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from PROXY.sectors.K_Agriculture.k_config import project_root

from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points
from .lucas_survey import norm_lucas_str


def point_rice_paddy_metric(row: Union[pd.Series, dict[str, Any]]) -> Optional[float]:
    """1.0 if SURVEY_LC1 is LUCAS rice (B17); excluded (None) otherwise."""
    if isinstance(row, pd.Series):
        lc1 = row["SURVEY_LC1"]
    else:
        lc1 = row.get("SURVEY_LC1")
    lc = norm_lucas_str(lc1)
    if lc.startswith("B17"):
        return 1.0
    return None


def _mu_scalar(x: Optional[float]) -> float:
    if x is None:
        return float("nan")
    return float(x)


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = [_mu_scalar(point_rice_paddy_metric(row)) for _, row in pts.iterrows()]
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
