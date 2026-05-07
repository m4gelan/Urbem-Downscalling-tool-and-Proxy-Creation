"""
Pollutant scores S_{n,c}^p = sum_s alpha_{p,s} * n_{n,c} * rho_{n,c}^s  (Methodology eq. 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _alpha_rows(alpha_cfg: dict[str, Any], pollutant_key: str) -> list[dict[str, Any]]:
    pol = alpha_cfg.get("pollutants") or {}
    return list(pol.get(pollutant_key) or [])


def compute_pollutant_score(
    extent_df: pd.DataFrame,
    rho_by_process: dict[str, pd.DataFrame],
    alpha_cfg: dict[str, Any],
    pollutant_key: str,
    score_col: str,
) -> pd.DataFrame:
    """
    extent_df: NUTS_ID, CLC_CODE, n_pixels, NAME_REGION, COUNTRY, ...
    rho_by_process[process_id]: columns NUTS_ID, CLC_CODE, rho
    """
    df = extent_df.copy()
    rows = _alpha_rows(alpha_cfg, pollutant_key)
    if not rows:
        df[score_col] = np.nan
        return df

    n = len(df)
    acc = np.zeros(n, dtype=float)
    for spec in rows:
        pid = str(spec.get("process_id", "")).strip()
        a = float(spec.get("alpha", 0.0))
        rdf = rho_by_process.get(pid)
        if rdf is None or rdf.empty:
            rho = np.zeros(n)
        else:
            m = df[["NUTS_ID", "CLC_CODE"]].merge(
                rdf[["NUTS_ID", "CLC_CODE", "rho"]],
                on=["NUTS_ID", "CLC_CODE"],
                how="left",
            )
            rho = m["rho"].fillna(0.0).to_numpy(dtype=float)
        acc += a * rho
    df[score_col] = df["n_pixels"].to_numpy(dtype=float) * acc
    return df


def merge_rho_lookup(
    extent_df: pd.DataFrame,
    rho_by_process: dict[str, pd.DataFrame],
    pollutant_key: str,
    alpha_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Attach combined alpha-weighted rho for diagnostics: sum_s alpha_s * rho_s."""
    df = extent_df.copy()
    rows = _alpha_rows(alpha_cfg, pollutant_key)
    n = len(df)
    acc = np.zeros(n, dtype=float)
    for spec in rows:
        pid = str(spec.get("process_id", "")).strip()
        a = float(spec.get("alpha", 0.0))
        rdf = rho_by_process.get(pid)
        if rdf is None or rdf.empty:
            rho = np.zeros(n)
        else:
            m = df[["NUTS_ID", "CLC_CODE"]].merge(
                rdf[["NUTS_ID", "CLC_CODE", "rho"]],
                on=["NUTS_ID", "CLC_CODE"],
                how="left",
            )
            rho = m["rho"].fillna(0.0).to_numpy(dtype=float)
        acc += a * rho
    df["rho_weighted"] = acc
    return df
