"""Pollutant scores for agriculture NUTS2 x CLC rows."""
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
