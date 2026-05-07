"""
Fertilized soil / synthetic N: LC1/LU1 eligibility + literature N rates -> mu per NUTS-2 x CLC -> rho.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY.sectors.K_Agriculture.k_config import project_root
from PROXY.core.dataloaders import resolve_path

from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .lucas_points import get_lucas_ag_points


def _norm(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().strip('"').strip("'").upper()
    t = " ".join(t.split())
    if t in ("NAN", "NONE", "NA", "#N/A", "NAT"):
        return ""
    return t


def load_synthetic_n_rates_json(path: Path) -> dict[str, float]:
    """Load `rates_and_LC` from YAML or JSON. Keys must be \"crop_label, LC1\" (LC1 = LUCAS code, e.g. B11, E10)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at root")
    rates = data.get("rates_and_LC") or {}
    if not rates and data.get("rates"):
        raise ValueError(
            f"{path}: expected 'rates_and_LC' with keys like 'wheat, B11'. Legacy 'rates' alone is no longer supported."
        )
    out: dict[str, float] = {}
    for k, v in rates.items():
        if str(k).startswith("_"):
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def synthetic_n_rate_for_lc1(lc1: Any, rates_and_lc: dict[str, float]) -> float | None:
    """Match SURVEY_LC1 to the LC segment after the last ', ' in each key (LUCAS: B=cropland, E=grassland)."""
    lc = _norm(lc1)
    if not lc:
        return None
    for key in sorted(rates_and_lc.keys()):
        if ", " not in key:
            continue
        _, lc_part = key.rsplit(", ", 1)
        if _norm(lc_part) == lc:
            return float(rates_and_lc[key])
    return None


def eligible_synthetic_n(lc1: Any, lu1: Any) -> bool:
    lc = _norm(lc1)
    lu = _norm(lu1)
    if not lc:
        return False
    if lc.startswith("B"):
        return True
    if lc == "E10":
        return True
    if lc == "E20" and lu == "U111":
        return True
    return False


def point_synth_n_rate(lc1: Any, lu1: Any, rates_and_lc: dict[str, float]) -> float:
    if not eligible_synthetic_n(lc1, lu1):
        return float("nan")
    v = synthetic_n_rate_for_lc1(lc1, rates_and_lc)
    if v is None:
        return float("nan")
    return float(v)


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    lb = cfg.get("lucas_build") or {}
    rate_rel = (
        lb.get("synthetic_n_rate")
        or lb.get("synthetic_n_rate_json")
        or "PROXY/config/agriculture/synthetic_N_rate.yaml"
    )
    rates_path = resolve_path(root, rate_rel)
    rates_and_lc = load_synthetic_n_rates_json(rates_path)

    pts = get_lucas_ag_points(cfg, root)
    pts = pts.copy()
    pts["mu"] = [
        point_synth_n_rate(lc, lu, rates_and_lc) for lc, lu in zip(pts["SURVEY_LC1"], pts["SURVEY_LU1"])
    ]
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
