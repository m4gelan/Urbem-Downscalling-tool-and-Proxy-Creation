"""
Biomass burning (field crop residues, NFR 4.F / GNFR L).

Step 1 — GFED4.1s: where agricultural fires occur (kg DM yr-1 per NUTS-2).
Step 2 — LUCAS: residue weight w_bar per (NUTS-2, CLC) from EMEP residue ratios × LC cover %.
Step 3 — Split mu_burn[n] across CLC using n_pixels * w_bar (ag classes 12–22 only).
Step 4 — rho = mu / max(mu) within country.

Prerequisites: run ``python -m Agriculture.preprocess.build_gfed41s_agri_mean`` once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Agriculture.config import project_root
from Agriculture.core.io import resolve_path

from .lucas_points import get_lucas_ag_points

_DEFAULT_DM_NPY = "data/Agriculture/gfed41s_agri_dm_mean.npy"
_DEFAULT_AREA_NPY = "data/Agriculture/gfed41s_grid_area.npy"
_DEFAULT_LOOKUP = "data/Agriculture/gfed41s_nuts2_lookup.parquet"
_PREPROCESS_CMD = "python -m Agriculture.preprocess.build_gfed41s_agri_mean"

# NFR 4.F eligible LUCAS LC1 codes: kg dry residue / kg grain yield (EMEP/EEA Guidebook Table 3-2)
RESIDUE_RATIOS: dict[str, float] = {
    "B11": 1.3,
    "B12": 1.3,
    "B13": 1.2,
    "B14": 1.6,
    "B15": 1.3,
    "B16": 1.0,
    "B17": 1.4,
    "B18": 1.2,
    "B19": 1.2,
    "B33": 2.1,
    "B41": 1.7,
}


def _norm_lc_code(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().upper()
    return s


def _perc_fraction(row: pd.Series, col: str) -> float:
    """SURVEY_LC*_PERC as fraction in [0,1]; missing -> 1.0."""
    if col not in row.index:
        return 1.0
    v = row[col]
    if pd.isna(v) or v == "":
        return 1.0
    try:
        return float(v) / 100.0
    except (TypeError, ValueError):
        return 1.0


def residue_score_per_point(row: pd.Series) -> float:
    """
    If SURVEY_LC1 in C_emep: RESIDUE_RATIOS[code] * (LC1_PERC/100 or 1.0).
    Elif SURVEY_LC2 in C_emep: same with LC2 and LC2_PERC.
    Else 0.
    """
    lc1 = _norm_lc_code(row.get("SURVEY_LC1"))
    lc2 = _norm_lc_code(row.get("SURVEY_LC2"))

    if lc1 in RESIDUE_RATIOS:
        return float(RESIDUE_RATIOS[lc1]) * _perc_fraction(row, "SURVEY_LC1_PERC")
    if lc2 in RESIDUE_RATIOS:
        return float(RESIDUE_RATIOS[lc2]) * _perc_fraction(row, "SURVEY_LC2_PERC")
    return 0.0


def aggregate_w_bar_lucas(points: pd.DataFrame) -> pd.DataFrame:
    """
    Per (NUTS_ID, CLC_CODE): N_nc = point count, w_bar = mean residue_score.
    Points with score 0 still count toward N_nc.
    """
    pts = points.copy()
    pts["residue_score"] = pts.apply(residue_score_per_point, axis=1)
    g = (
        pts.groupby(["NUTS_ID", "CLC_CODE"], as_index=False)
        .agg(
            n_lucas_points=("residue_score", "size"),
            sum_w=("residue_score", "sum"),
        )
    )
    g["w_bar"] = np.where(
        g["n_lucas_points"] > 0,
        g["sum_w"].astype(np.float64) / g["n_lucas_points"].astype(np.float64),
        0.0,
    )
    return g[["NUTS_ID", "CLC_CODE", "n_lucas_points", "w_bar"]]


def pixel_agri_dm_score(dm_mean: np.ndarray, area_m2: np.ndarray) -> np.ndarray:
    """Per-pixel annual agricultural DM burned (kg DM yr-1), shape (720, 1440)."""
    return (dm_mean * area_m2).astype(np.float64)


def aggregate_to_nuts2(pixel_score: np.ndarray, lookup: pd.DataFrame) -> pd.Series:
    """Sum pixel scores per NUTS-2; index NUTS_ID, values kg DM yr-1."""
    flat = pixel_score.ravel()
    lk = lookup.copy()
    lk["score"] = flat[lk["lat_idx"].to_numpy(np.int64) * 1440 + lk["lon_idx"].to_numpy(np.int64)]
    return lk.groupby("NUTS_ID")["score"].sum()


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Biomass burning proxy: mu and rho per (NUTS_ID, CLC_CODE).

    Returns columns: NUTS_ID, CLC_CODE, COUNTRY, mu, rho, n_lucas_points, w_bar
    """
    root = project_root(cfg)
    lb = cfg.get("lucas_build") or {}

    dm_path = resolve_path(root, lb.get("gfed41s_agri_dm_mean_npy", _DEFAULT_DM_NPY))
    area_path = resolve_path(root, lb.get("gfed41s_grid_area_npy", _DEFAULT_AREA_NPY))
    lookup_path = resolve_path(root, lb.get("gfed41s_nuts2_lookup_parquet", _DEFAULT_LOOKUP))

    for p in (dm_path, area_path, lookup_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"GFED4.1s preprocessing artefact not found: {p}\n"
                f"Run: {_PREPROCESS_CMD}"
            )

    dm_mean = np.load(dm_path).astype(np.float64)
    area_m2 = np.load(area_path).astype(np.float64)
    lookup = pd.read_parquet(lookup_path)

    scores = pixel_agri_dm_score(dm_mean, area_m2)
    mu_burn = aggregate_to_nuts2(scores, lookup)

    lucas_pts = get_lucas_ag_points(cfg, root)
    wbar_df = aggregate_w_bar_lucas(lucas_pts)

    ext = extent_df[["NUTS_ID", "CLC_CODE", "COUNTRY", "n_pixels"]].copy()
    merged = ext.merge(
        wbar_df,
        on=["NUTS_ID", "CLC_CODE"],
        how="left",
    )
    merged["w_bar"] = merged["w_bar"].fillna(0.0)
    merged["n_lucas_points"] = merged["n_lucas_points"].fillna(0).astype(np.int64)

    merged["weighted_pix"] = merged["n_pixels"].astype(np.float64) * merged["w_bar"]
    denom_by_nuts = merged.groupby("NUTS_ID", sort=False)["weighted_pix"].sum()
    mu_series = mu_burn.astype(np.float64)
    mu_series.index = mu_series.index.astype(str).str.strip()
    merged["mu_burn_n"] = merged["NUTS_ID"].astype(str).str.strip().map(mu_series).fillna(0.0)
    denom_row = merged["NUTS_ID"].map(denom_by_nuts.to_dict()).fillna(0.0)

    ok = (denom_row > 0) & (merged["mu_burn_n"] > 0)
    merged["mu"] = np.where(
        ok,
        merged["mu_burn_n"] * merged["weighted_pix"] / denom_row,
        0.0,
    )

    max_mu = merged.groupby("COUNTRY")["mu"].transform("max")
    merged["rho"] = np.where(max_mu > 0, (merged["mu"] / max_mu).clip(0.0, 1.0), 0.0)

    out = merged[
        ["NUTS_ID", "CLC_CODE", "COUNTRY", "mu", "rho", "n_lucas_points", "w_bar"]
    ].copy()
    return out
