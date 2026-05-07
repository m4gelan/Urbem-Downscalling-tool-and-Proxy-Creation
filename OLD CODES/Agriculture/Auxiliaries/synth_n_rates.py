"""
Synthetic mineral N application rates (kg N ha⁻¹ yr⁻¹) derived from:
Einarsson, R., Sanz-Cobena, A., Aguilera, E. et al. (2022), "Crop production and nitrogen use in European cropland and grassland 1961–2019", Scientific Data, https://www.nature.com/articles/s41597-021-01061

Data accessed from the authors' model and data archive: https://figshare.com/articles/dataset/EuropeAgriDB_v1_0_Source_code_and_data/16540314

Source files:
  - model/outdata/D/literature_data_n_aggregated.csv
  - model/outdata/D/literature_data_n_inputs.csv

See script and dataset for additional details.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_AUX_DIR = Path(__file__).resolve().parent
# Crop-level mineral N (has Nitrogen flow, Crop, Rate): do not swap with aggregated.
_LIT_CSV = _AUX_DIR / "literature_data_n_inputs.csv"
_AGG_CSV = _AUX_DIR / "literature_data_aggregated.csv"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _map_crop_to_bucket(crop: str) -> str | None:
    c = _norm(crop)
    rules: list[tuple[callable, str]] = [
        (lambda x: x.startswith("wheat"), "wheat"),
        (
            lambda x: "grain maize" in x
            or x.startswith("grain maize")
            or x == "maize (for corn)"
            or x == "corn",
            "maize_grain",
        ),
        (lambda x: x.startswith("barley"), "barley"),
        (
            lambda x: x.startswith("rye") and "oat" not in x and "rice" not in x,
            "rye",
        ),
        (lambda x: x == "oat", "oats"),
        (lambda x: "rye, oat, rice" in x, "rye_oat_rice_mix"),
        (lambda x: x.startswith("triticale"), "other_cereal"),
        (
            lambda x: "buckwheat" in x
            or "cereals mixture" in x
            or x in ("other crops", "other"),
            "other_cereal",
        ),
        (lambda x: x.startswith("potato"), "potato"),
        (lambda x: "sugar beet" in x, "sugar_beet"),
        (
            lambda x: x in ("sunflower", "sunflower seed")
            or (x.startswith("sunflower") and "soya" not in x),
            "sunflower",
        ),
        (
            lambda x: "oilseed rape" in x
            or x in ("rapeseed", "rape")
            or "rape seed" in x
            or x == "mustard"
            or x == "oilseed (rape)",
            "rapeseed",
        ),
        (lambda x: "soybean" in x or x == "soya", "soya"),
        (lambda x: "sunflower, soya, linseed" in x, "oilseed_mixed"),
        (lambda x: x == "tobacco", "tobacco"),
        (
            lambda x: "set-aside" in x
            or "industrial crops" in x
            or x == "other (including tobacco)",
            "other_industrial",
        ),
        (lambda x: x.startswith("pulses") or x == "pea", "dry_pulse"),
        (lambda x: x in ("vegetables", "field vegetables"), "other_veg"),
        (lambda x: "maize, silage" in x, "fodder_maize_silage"),
        (
            lambda x: x
            in (
                "alfalfa",
                "clover",
                "fodder (legume)",
                "fodder (legumes)",
                "fodder (pulses)",
                "leguminous, grasses",
            ),
            "fodder_legume",
        ),
        (lambda x: "fodder" in x and "beet" in x, "fodder_beet"),
        (
            lambda x: "green fodder" in x
            or x.startswith("fodder (other)")
            or "fodder, perennial (other)" in x,
            "fodder_other",
        ),
        (lambda x: x == "vineyard" or x == "grape", "vineyard"),
        (
            lambda x: "permanent crops" in x or "fruit, vineyard" in x,
            "permanent_crops_mixed",
        ),
        (lambda x: x in ("fruits", "orchards", "apple"), "fruit_tree"),
        (
            lambda x: "grassland" in x and "fertilized" in x,
            "temporary_grass",
        ),
        (
            lambda x: x in ("grassland", "pasture", "meadows (for hay)"),
            "permanent_grass_proxy",
        ),
        (lambda x: x in ("flax", "fibre flax", "hop", "poppy"), "other_industrial"),
    ]
    for pred, key in rules:
        if pred(c):
            return key
    return None


def _agg_region_bucket(g: pd.DataFrame) -> float:
    a = g["A"].to_numpy()
    r = g["R"].to_numpy()
    ta = float(np.nansum(a))
    if ta > 0:
        return float(np.nansum(r * a) / ta)
    return float(np.nanmean(r))


def _eu_area_weighted(rpk: pd.DataFrame, bucket: str) -> float:
    d = rpk.loc[rpk["k"] == bucket, ["A", "R"]]
    if d.empty:
        return float("nan")
    a = d["A"].to_numpy()
    r = d["R"].to_numpy()
    ta = float(np.nansum(a))
    if ta > 0:
        return float(np.nansum(r * a) / ta)
    return float(np.nanmean(r))


def _fao_crop_table_mineral_2000() -> pd.DataFrame:
    df = pd.read_csv(_LIT_CSV)
    sub = df.loc[
        (df["Publication"] == "FAO et al. (2002)")
        & (df["Nitrogen flow"] == "Mineral")
        & df["Crop"].notna()
        & (df["Year"] == 2000)
    ].copy()
    sub = sub.loc[sub["Region"] != "Switzerland"]
    sub["A"] = pd.to_numeric(sub["Area (Mha)"], errors="coerce").fillna(0.0)
    sub["R"] = pd.to_numeric(sub["Rate (kg N/ha)"], errors="coerce")
    sub = sub.loc[sub["R"].notna()]
    sub["k"] = sub["Crop"].map(_map_crop_to_bucket)
    if sub["k"].isna().any():
        bad = sub.loc[sub["k"].isna(), "Crop"].unique()
        raise ValueError(f"Unmapped FAO 2000 crops: {bad!r}")
    return sub


def _region_bucket_rates(sub: pd.DataFrame) -> pd.DataFrame:
    return (
        sub.groupby(["Region", "k"], sort=True)
        .apply(_agg_region_bucket, include_groups=False)
        .reset_index()
        .rename(columns={0: "R"})
        .merge(
            sub.groupby(["Region", "k"], sort=True)["A"].sum().reset_index(),
            on=["Region", "k"],
        )
    )


def _rice_area_weighted_fao() -> float:
    df = pd.read_csv(_LIT_CSV)
    rice = df.loc[
        (df["Nitrogen flow"] == "Mineral")
        & (df["Crop"].astype(str).str.strip() == "Rice")
        & df["Publication"].astype(str).str.contains("FAO", na=False)
    ].copy()
    rice["A"] = pd.to_numeric(rice["Area (Mha)"], errors="coerce").fillna(0.0)
    rice["R"] = pd.to_numeric(rice["Rate (kg N/ha)"], errors="coerce")
    rice = rice.loc[rice["R"].notna()]
    if rice.empty:
        return float("nan")
    a = rice["A"].to_numpy()
    r = rice["R"].to_numpy()
    ta = float(np.nansum(a))
    if ta > 0:
        return float(np.nansum(r * a) / ta)
    return float(np.nanmean(r))


def _single_crop_mean_fao(crop_label: str) -> float:
    df = pd.read_csv(_LIT_CSV)
    m = df.loc[
        (df["Nitrogen flow"] == "Mineral")
        & (df["Crop"].astype(str).str.strip() == crop_label)
        & df["Publication"].astype(str).str.contains("FAO", na=False)
    ]
    r = pd.to_numeric(m["Rate (kg N/ha)"], errors="coerce").dropna()
    if r.empty:
        return float("nan")
    return float(r.mean())


def _set_aside_industrial_only_eu_2000() -> float:
    df = pd.read_csv(_LIT_CSV)
    sub = df.loc[
        (df["Publication"] == "FAO et al. (2002)")
        & (df["Nitrogen flow"] == "Mineral")
        & (df["Year"] == 2000)
        & (df["Region"] != "Switzerland")
        & df["Crop"].astype(str).str.contains("Set-aside", case=False, na=False)
    ].copy()
    sub["A"] = pd.to_numeric(sub["Area (Mha)"], errors="coerce").fillna(0.0)
    sub["R"] = pd.to_numeric(sub["Rate (kg N/ha)"], errors="coerce")
    sub = sub.loc[sub["R"].notna()]
    a = sub["A"].to_numpy()
    r = sub["R"].to_numpy()
    ta = float(np.nansum(a))
    if ta > 0:
        return float(np.nansum(r * a) / ta)
    return float("nan")


def _efma_2006_pg_area_weighted() -> float:
    agg = pd.read_csv(_AGG_CSV)
    efma = agg.loc[
        agg["Publication"].astype(str).str.contains("EFMA", na=False)
        & (agg["Year"] == 2006)
    ].copy()
    efma["A"] = pd.to_numeric(efma["A"], errors="coerce")
    efma["R"] = pd.to_numeric(efma["R"], errors="coerce")
    pg = efma.loc[
        efma["Interpretation"].astype(str).str.match("PG", na=False)
        & (efma["A"] > 0)
        & efma["R"].notna()
    ]
    if pg.empty:
        return float("nan")
    return float(np.average(pg["R"].to_numpy(), weights=pg["A"].to_numpy()))


def compute_synth_n_rates() -> dict[str, float]:
    """
    Return synthetic N application rates (kg N ha-1 yr-1) as floats before rounding.
    """
    sub = _fao_crop_table_mineral_2000()
    rpk = _region_bucket_rates(sub)

    fodder_keys = (
        "fodder_maize_silage",
        "fodder_beet",
        "fodder_legume",
        "fodder_other",
    )
    fd = rpk.loc[rpk["k"].isin(fodder_keys), ["A", "R"]]
    fodder_r = (
        float(np.sum(fd["R"].to_numpy() * fd["A"].to_numpy()) / float(np.sum(fd["A"])))
        if len(fd) and float(np.sum(fd["A"])) > 0
        else float("nan")
    )

    return {
        "wheat": _eu_area_weighted(rpk, "wheat"),
        "maize_grain": _eu_area_weighted(rpk, "maize_grain"),
        "rice": _rice_area_weighted_fao(),
        "barley": _eu_area_weighted(rpk, "barley"),
        "rye": _eu_area_weighted(rpk, "rye"),
        "oats": _eu_area_weighted(rpk, "oats"),
        "other_cereal": _eu_area_weighted(rpk, "other_cereal"),
        "potato": _eu_area_weighted(rpk, "potato"),
        "sugar_beet": _eu_area_weighted(rpk, "sugar_beet"),
        "sunflower": _eu_area_weighted(rpk, "sunflower"),
        "rapeseed": _eu_area_weighted(rpk, "rapeseed"),
        "soya": _eu_area_weighted(rpk, "soya"),
        "cotton": _set_aside_industrial_only_eu_2000(),
        "tobacco": _eu_area_weighted(rpk, "tobacco"),
        "other_industrial": _eu_area_weighted(rpk, "other_industrial"),
        "dry_pulse": _eu_area_weighted(rpk, "dry_pulse"),
        "tomato": _single_crop_mean_fao("Tomato"),
        "other_veg": _eu_area_weighted(rpk, "other_veg"),
        "fodder_crop": fodder_r,
        "olive": _single_crop_mean_fao("Olive"),
        "vineyard": _eu_area_weighted(rpk, "vineyard"),
        "fruit_tree": _eu_area_weighted(rpk, "fruit_tree"),
        "citrus": _eu_area_weighted(rpk, "fruit_tree"),
        "nuts_tree": _eu_area_weighted(rpk, "permanent_crops_mixed"),
        "temporary_grass": _eu_area_weighted(rpk, "temporary_grass"),
        "permanent_grass": _efma_2006_pg_area_weighted(),
        "shrub_grass": 10.0,
        "fallow": 0.0,
    }


def _rounded_int_rates() -> dict[str, int]:
    raw = compute_synth_n_rates()
    out: dict[str, int] = {}
    for k, v in raw.items():
        if k == "fallow":
            out[k] = 0
            continue
        if v != v:
            out[k] = 0
        else:
            out[k] = int(round(float(v)))
    return out


SYNTH_N_RATE: dict[str, int] = _rounded_int_rates()
print(SYNTH_N_RATE)
__all__ = ["SYNTH_N_RATE", "compute_synth_n_rates"]
