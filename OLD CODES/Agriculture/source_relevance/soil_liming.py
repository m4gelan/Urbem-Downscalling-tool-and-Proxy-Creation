"""
Soil liming (NFR 3.G.1 CO2): LUCAS Soil 2018 pH_H2O + OC + LC, RB209 piecewise lime demand (Goulding 2016).

Loads LUCAS-SOIL-2018.csv, samples CORINE at TH_LAT/TH_LONG, aggregates mean mu per (NUTS-2, CLC).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Agriculture.config import project_root
from Agriculture.core.io import resolve_path
from Agriculture.core.run_countries import parse_run_country_codes

from .common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from .lucas_points import sample_corine_clc

_ORG_OC_G_KG = 60.0


def _clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"< LOD": np.nan, "<lod": np.nan, "nan": np.nan, "": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _norm_lc(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    t = str(val).strip().upper()
    return t


def lime_demand_min_ar(ph: float) -> float:
    """Mineral arable: Eq. lime-min-ar (methods.tex). ph = pH_H2O."""
    if ph >= 6.5:
        return 0.0
    if ph >= 6.0:
        return 4.5 * (6.5 - ph) / 0.5
    if ph >= 5.5:
        return 4.5 + 4.5 * (6.0 - ph) / 0.5
    if ph >= 5.0:
        return 9.0 + 4.0 * (5.5 - ph) / 0.5
    return 13.0


def lime_demand_min_gr(ph: float) -> float:
    """Mineral grass: Eq. lime-min-gr."""
    if ph >= 5.5:
        return 0.0
    if ph >= 5.0:
        return 3.5 * (5.5 - ph) / 0.5
    if ph >= 4.5:
        return 3.5 + 3.0 * (5.0 - ph) / 0.5
    return 6.5


def lime_demand_org_ar(ph: float) -> float:
    """Organic/peaty arable: Eq. lime-org-ar."""
    if ph >= 6.0:
        return 0.0
    if ph >= 5.5:
        return 8.5 * (6.0 - ph) / 0.5
    if ph >= 5.0:
        return 8.5 + 7.0 * (5.5 - ph) / 0.5
    return 15.5


def lime_demand_org_gr(ph: float) -> float:
    """Organic/peaty grass: Eq. lime-org-gr."""
    if ph >= 5.5:
        return 0.0
    if ph >= 5.0:
        return 1.5 * (5.5 - ph) / 0.5
    if ph >= 4.5:
        return 1.5 + 5.0 * (5.0 - ph) / 0.5
    return 6.5


def lime_score(ph_h2o: float, oc_g_kg: float, lc: str) -> float:
    """
    s_p^lime in t CaCO3 ha-1 yr-1 (demand proxy). Missing pH -> NaN.
    OC missing -> treated as mineral (OC <= threshold).
    """
    if ph_h2o is None or (isinstance(ph_h2o, float) and np.isnan(ph_h2o)):
        return float("nan")
    ph = float(ph_h2o)
    oc = float(oc_g_kg) if oc_g_kg is not None and not (isinstance(oc_g_kg, float) and np.isnan(oc_g_kg)) else 0.0
    organic = oc > _ORG_OC_G_KG
    lc_n = _norm_lc(lc)
    arable = lc_n.startswith("B")
    grass = lc_n in ("E10", "E20")
    if not arable and not grass:
        return float("nan")
    if organic:
        return lime_demand_org_ar(ph) if arable else lime_demand_org_gr(ph)
    return lime_demand_min_ar(ph) if arable else lime_demand_min_gr(ph)


def load_lucas_soil_liming_points(cfg: dict[str, Any], root: Any) -> pd.DataFrame:
    """
    LUCAS Soil 2018 rows: agricultural LC, ag CLC from raster, NUTS_ID, COUNTRY, mu = lime score.
    """
    lb = cfg.get("lucas_build") or {}
    paths = cfg.get("paths") or {}
    inputs = paths.get("inputs") or {}
    run = cfg.get("run") or {}
    nodata = float(run.get("nodata", -128.0))
    country_codes = parse_run_country_codes(run)

    soil_rel = lb.get("lucas_soil_2018_csv") or "data/Agriculture/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv"
    soil_path = resolve_path(root, soil_rel)
    corine_rel = lb.get("corine_raster") or inputs.get("corine_raster", "Input/CORINE/U2018_CLC2018_V2020_20u1.tif")
    corine_path = resolve_path(root, corine_rel)

    if not soil_path.is_file():
        raise FileNotFoundError(f"LUCAS Soil 2018 CSV not found: {soil_path}")
    if not corine_path.is_file():
        raise FileNotFoundError(f"CORINE raster not found: {corine_path}")

    usecols = [
        "TH_LAT",
        "TH_LONG",
        "NUTS_0",
        "NUTS_2",
        "LC",
        "pH_H2O",
        "OC",
    ]
    raw = pd.read_csv(soil_path, usecols=lambda c: c in usecols, low_memory=False)
    for c in usecols:
        if c not in raw.columns:
            raise ValueError(f"LUCAS Soil CSV missing column {c!r}: {soil_path}")

    raw["pH_H2O"] = _clean_numeric(raw["pH_H2O"])
    raw["OC"] = _clean_numeric(raw["OC"])
    raw["LC"] = raw["LC"].map(_norm_lc)
    raw["_lat"] = pd.to_numeric(raw["TH_LAT"], errors="coerce")
    raw["_lon"] = pd.to_numeric(raw["TH_LONG"], errors="coerce")
    raw = raw.dropna(subset=["_lat", "_lon"]).copy()

    lc = raw["LC"].astype(str)
    eligible = lc.str.startswith("B") | lc.isin(["E10", "E20"])
    raw = raw.loc[eligible].copy()

    raw["NUTS_ID"] = raw["NUTS_2"].astype(str).str.strip()
    raw["COUNTRY"] = raw["NUTS_0"].astype(str).str.strip().str.upper()
    if country_codes:
        raw = raw[raw["COUNTRY"].isin(country_codes)].copy()
        if raw.empty:
            raise ValueError(f"No LUCAS Soil rows for country in {sorted(country_codes)!r}.")

    lat = raw["_lat"].to_numpy(dtype=float)
    lon = raw["_lon"].to_numpy(dtype=float)
    clc = sample_corine_clc(lon, lat, corine_path, nodata=nodata)
    raw["CLC_CODE"] = np.asarray(clc, dtype=float)
    raw = raw[np.isfinite(raw["CLC_CODE"])].copy()
    raw["CLC_CODE"] = raw["CLC_CODE"].astype(int)

    ag_clc = tuple(range(12, 23))
    raw = raw[raw["CLC_CODE"].isin(ag_clc)].copy()
    if raw.empty:
        raise ValueError("No LUCAS Soil points on agricultural CLC (12–22) after CORINE sampling.")

    raw["mu"] = [
        lime_score(ph, oc, lc)
        for ph, oc, lc in zip(raw["pH_H2O"], raw["OC"], raw["LC"].astype(str))
    ]
    raw = raw[np.isfinite(raw["mu"])].copy()
    if raw.empty:
        raise ValueError("No LUCAS Soil points with valid pH_H2O lime score.")

    return raw[["NUTS_ID", "CLC_CODE", "COUNTRY", "mu"]]


def compute_rho_df(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    root = project_root(cfg)
    pts = load_lucas_soil_liming_points(cfg, root)
    agg = aggregate_nuts_clc_mu(pts, "mu")
    return merge_extent_mu_rho(extent_df, agg)[["NUTS_ID", "CLC_CODE", "mu", "rho"]]
