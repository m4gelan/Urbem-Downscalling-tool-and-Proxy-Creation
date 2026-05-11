"""Console-friendly CEIP / alpha summaries for sector builds (no extra artifacts)."""

from __future__ import annotations

import logging

import pandas as pd


def log_wide_group_alpha_table(
    logger: logging.Logger,
    *,
    sector: str,
    wide: pd.DataFrame,
    focus_iso3: str,
    group_cols: tuple[str, ...] = ("alpha_G1", "alpha_G2", "alpha_G3", "alpha_G4"),
) -> None:
    """Log ``wide`` rows for one ``country_iso3`` (GNFR B/D style CEIP groups)."""
    iso = str(focus_iso3).strip().upper()
    if wide.empty or "country_iso3" not in wide.columns:
        logger.info("%s CEIP alpha table: (empty wide frame)", sector)
        return
    sub = wide.loc[wide["country_iso3"].astype(str).str.upper() == iso].copy()
    if sub.empty:
        logger.info("%s CEIP alpha table: no rows for country_iso3=%s", sector, iso)
        return
    cols = ["pollutant", *group_cols]
    if "method" in sub.columns:
        cols.append("method")
    elif "fallback_code" in sub.columns:
        cols.append("fallback_code")
    use = [c for c in cols if c in sub.columns]
    block = sub[use].to_string(index=False)
    logger.info("%s CEIP alpha decomposition (country_iso3=%s):\n%s", sector, iso, block)


def log_waste_family_weights(
    logger: logging.Logger,
    *,
    sector: str,
    wide: pd.DataFrame,
    focus_iso3: str,
) -> None:
    iso = str(focus_iso3).strip().upper()
    if wide.empty or "country_iso3" not in wide.columns:
        logger.info("%s CEIP family weights: (empty wide frame)", sector)
        return
    sub = wide.loc[wide["country_iso3"].astype(str).str.upper() == iso].copy()
    if sub.empty:
        logger.info("%s CEIP family weights: no rows for country_iso3=%s", sector, iso)
        return
    cols = [
        c
        for c in (
            "pollutant",
            "w_solid",
            "w_ww",
            "w_res",
            "method",
            "fallback_tier",
        )
        if c in sub.columns
    ]
    block = sub[cols].to_string(index=False)
    logger.info("%s CEIP family weights (country_iso3=%s):\n%s", sector, iso, block)


def log_solvent_alpha_pivot(
    logger: logging.Logger,
    *,
    sector: str,
    agg: pd.DataFrame,
    focus_iso3: str,
    subsectors: list[str],
) -> None:
    iso = str(focus_iso3).strip().upper()
    if agg.empty:
        logger.info("%s CEIP alpha pivot: (empty agg)", sector)
        return
    need = {"country_iso3", "pollutant", "subsector", "alpha"}
    if not need.issubset(agg.columns):
        logger.info("%s CEIP alpha pivot: missing columns (need %s)", sector, sorted(need))
        return
    sub = agg.loc[agg["country_iso3"].astype(str).str.upper() == iso].copy()
    if sub.empty:
        logger.info("%s CEIP alpha pivot: no rows for country_iso3=%s", sector, iso)
        return
    pt = sub.pivot_table(
        index="pollutant",
        columns="subsector",
        values="alpha",
        aggfunc="first",
    )
    for s in subsectors:
        if s not in pt.columns:
            pt[s] = float("nan")
    pt = pt.reindex(columns=[c for c in subsectors if c in pt.columns], fill_value=float("nan"))
    block = pt.to_string(float_format=lambda x: f"{x:.6g}")
    logger.info("%s CEIP alpha decomposition (country_iso3=%s):\n%s", sector, iso, block)
