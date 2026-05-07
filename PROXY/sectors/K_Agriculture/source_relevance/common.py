"""
Shared helpers: raw mu from CLC maps or LUCAS aggregates, then rho = mu / max mu within country (eq. 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rho_by_country_group(mu: pd.Series) -> pd.Series:
    """Normalize mu within each country group (same index as mu)."""
    m = float(mu.max())
    if m <= 0 or np.isnan(m):
        return pd.Series(0.0, index=mu.index)
    return mu / m


def aggregate_nuts_clc_mu(points: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Mean of value_col per (NUTS_ID, CLC_CODE, COUNTRY), then rho = mu / max mu per country.
    Returns columns: NUTS_ID, CLC_CODE, COUNTRY, mu, rho.
    """
    g = points.groupby(["NUTS_ID", "CLC_CODE", "COUNTRY"], as_index=False).agg(mu=(value_col, "mean"))
    g["mu"] = g["mu"].fillna(0.0)
    g["rho"] = g.groupby("COUNTRY")["mu"].transform(rho_by_country_group)
    return g


def apply_census_omega_to_agg(
    agg: pd.DataFrame,
    omega_by_nuts: pd.Series,
    missing_fallback: float = 1.0,
) -> pd.DataFrame:
    """Multiply mu by omega(NUTS_ID), then recompute rho within country."""
    agg = agg.copy()
    idx = pd.Index(omega_by_nuts.index.astype(str).str.strip())
    omega2 = pd.Series(omega_by_nuts.values, index=idx)
    w = agg["NUTS_ID"].astype(str).str.strip().map(omega2).fillna(missing_fallback)
    agg["mu"] = agg["mu"].to_numpy(dtype=float) * w.to_numpy(dtype=float)
    agg["rho"] = agg.groupby("COUNTRY")["mu"].transform(rho_by_country_group)
    return agg


def merge_extent_mu_rho(extent_df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Left-merge extent rows with NUTS x CLC mu/rho; missing -> 0."""
    sub = agg[["NUTS_ID", "CLC_CODE", "mu", "rho"]].drop_duplicates(
        subset=["NUTS_ID", "CLC_CODE"], keep="first"
    )
    out = extent_df[["NUTS_ID", "CLC_CODE", "COUNTRY"]].merge(
        sub,
        on=["NUTS_ID", "CLC_CODE"],
        how="left",
    )
    out["mu"] = out["mu"].fillna(0.0)
    out["rho"] = out["rho"].fillna(0.0)
    return out


def rho_from_clc_mu(extent_df: pd.DataFrame, clc_mu: dict[int, float]) -> pd.DataFrame:
    """
    extent_df must include NUTS_ID, CLC_CODE, COUNTRY.
    Returns NUTS_ID, CLC_CODE, mu, rho with rho in [0, 1] per country.
    """
    df = extent_df[["NUTS_ID", "CLC_CODE", "COUNTRY"]].copy()
    df["mu"] = df["CLC_CODE"].map(lambda c: float(clc_mu.get(int(c), 0.0)))
    max_mu = df.groupby("COUNTRY")["mu"].transform("max")
    df["rho"] = np.where(max_mu > 0, df["mu"] / max_mu, 0.0)
    return df
