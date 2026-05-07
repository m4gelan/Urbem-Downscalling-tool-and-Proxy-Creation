"""
Livestock housing (NH3, GNFR K): LUCAS LU1/LC1 stage-1 score, census omega stage-2 -> mu per NUTS-2 x CLC -> rho.

LUCAS 2022 codes used here:
- U111 = Agriculture (non-fallow), U112 = fallow land, U113 = kitchen gardens (excluded).
- U120 = forest (user-confirmed 2026-04). Treated as excluded by default exactly like
  the legacy code. A YAML switch in ``PROXY/config/agriculture/lu1_lc1_mapping.yaml``
  (``livestock_housing.u120_mixed_livestock``) lets advanced users re-enable U120 as a
  livestock-housing-bearing class for research scenarios without editing Python, but it
  ships disabled and should stay that way for production runs.

Emission factors and C21 column names load from ``PROXY/config/agriculture/emission_factors.yaml``.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import yaml

from PROXY.sectors.K_Agriculture.tabular.emission_factors import load_livestock_housing_nh3
from PROXY.sectors.K_Agriculture.k_config import project_root

from .census_intensity import load_census_omega_for_cfg
from .common import aggregate_nuts_clc_mu, apply_census_omega_to_agg, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points
from .lucas_survey import norm_lucas_str

logger = logging.getLogger(__name__)

# Phase 1.5: tier values below mirror the JSON defaults and are used only if the JSON
# is missing (the loader already returns those same defaults). Keeping them as module
# constants preserves any code that imported them directly.
_HOUSING_EF = load_livestock_housing_nh3()
MISSING_G = _HOUSING_EF["missing_grazing_damping"]

_AGRI_LU_FALLBACK = frozenset({"U111", "U112"})
_BUILDING_LC = frozenset({"A11", "A12"})
_GRASSLAND_LC_PREFIX = "E"
_CROPLAND_LC_PREFIX = "B"
_LEY_LC = "B55"

_U120_YAML_RELPATH = Path("PROXY") / "config" / "agriculture" / "lu1_lc1_mapping.yaml"
_U120_LEGACY_BEHAVIOR = {"enabled": False, "score": None, "source": "legacy forestry exclusion"}


@lru_cache(maxsize=8)
def _load_u120_config(yaml_path: str | None = None) -> dict[str, Any]:
    """Load the U120 tier config from the LC1/LU1 mapping YAML.

    Results are cached per path so repeated calls (one per LUCAS point) are cheap.
    Missing file or missing keys fall back to the legacy forestry exclusion with a
    one-time WARNING so the change is never silent.
    """
    if yaml_path is None:
        candidates = [
            Path(__file__).resolve().parents[4] / _U120_YAML_RELPATH,
            Path.cwd() / _U120_YAML_RELPATH,
        ]
        for p in candidates:
            if p.is_file():
                yaml_path = str(p)
                break
        if yaml_path is None:
            logger.warning(
                "livestock_housing: could not locate %s; reverting to legacy U120 forestry exclusion.",
                _U120_YAML_RELPATH,
            )
            return dict(_U120_LEGACY_BEHAVIOR)
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "livestock_housing: failed to read %s (%s); reverting to legacy U120 forestry exclusion.",
            yaml_path,
            exc,
        )
        return dict(_U120_LEGACY_BEHAVIOR)

    block = ((data.get("livestock_housing") or {}).get("u120_mixed_livestock") or {})
    enabled = bool(block.get("enabled", False))
    score = block.get("score")
    if enabled and (score is None or not np.isfinite(float(score))):
        logger.warning(
            "livestock_housing: u120_mixed_livestock.enabled=true but score is missing/invalid in %s; "
            "reverting to legacy U120 forestry exclusion.",
            yaml_path,
        )
        return dict(_U120_LEGACY_BEHAVIOR)

    if enabled:
        logger.warning(
            "livestock_housing: U120 reinterpretation ACTIVE (non-default, NOT production), "
            "tier=%.3f (source=%s). This changes rho on any LUCAS point with LU1=U120; "
            "production default is u120_mixed_livestock.enabled=false in %s.",
            float(score),
            block.get("source", "(unspecified)"),
            _U120_YAML_RELPATH,
        )
    return {
        "enabled": enabled,
        "score": float(score) if score is not None else None,
        "source": block.get("source", ""),
    }


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

    U120 handling: the production default treats U120 as forest and excludes it
    (returns ``None``), matching the legacy behaviour. The YAML switch
    ``livestock_housing.u120_mixed_livestock.enabled`` lets advanced users opt in
    to a research-mode "mixed farming" score, but this is **not** the default.
    """
    lc = norm_lucas_str(row.get("SURVEY_LC1", ""))  # type: ignore[union-attr]
    lu = norm_lucas_str(row.get("SURVEY_LU1", ""))  # type: ignore[union-attr]

    if lu == "U113":
        return None

    u120_cfg = _load_u120_config()
    if lu == "U120" and not u120_cfg["enabled"]:
        return None

    g = _parse_grazing_code(row)

    if g == 1:
        return 0.0

    confirmed = g is not None and g in (0, 2)

    if lc in _BUILDING_LC and lu == "U111":
        tier = _HOUSING_EF["u111_building"]
    elif lu == "U111" and lc.startswith(_GRASSLAND_LC_PREFIX):
        tier = _HOUSING_EF["u111_grassland"]
    elif lu == "U111" and lc.startswith(_CROPLAND_LC_PREFIX) and lc != _LEY_LC:
        tier = _HOUSING_EF["u111_cropland_generic"]
    elif lu == "U111" and lc == _LEY_LC:
        tier = _HOUSING_EF["u111_ley"]
    elif lu == "U112":
        tier = _HOUSING_EF["u112_fallow"]
    elif lu == "U120" and u120_cfg["enabled"]:
        tier = float(u120_cfg["score"])
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

    u120_cfg = _load_u120_config()
    if u120_cfg["enabled"] and "SURVEY_LU1" in pts.columns:
        n_u120 = int(
            pts["SURVEY_LU1"].astype(str).str.upper().str.strip().eq("U120").sum()
        )
        if n_u120:
            logger.info(
                "livestock_housing: U120 reinterpretation applied to %d LUCAS point(s) "
                "(score=%.3f).",
                n_u120,
                float(u120_cfg["score"]),
            )

    agg = aggregate_nuts_clc_mu(pts, "mu")
    omega, fb = load_census_omega_for_cfg(cfg, root, "housing")
    agg = apply_census_omega_to_agg(agg, omega, fb)
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
