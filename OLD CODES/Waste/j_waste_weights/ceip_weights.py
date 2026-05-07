"""
CEIP / NRD national totals -> family weights (w_solid, w_ww, w_res) per country and pollutant.

Excludes incineration SNAP rows by default (see ``ceip_sector_codes.yaml``).
Fallback ladder when E_solid+E_ww+E_res == 0 for a country–pollutant pair:
  tier1 — mean family shares over other pollutants in the same country with positive totals
  tier2 — mean over EU countries with valid shares for that pollutant
  tier3 — equal weights (1/3, 1/3, 1/3)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config_loader import resolve_path

logger = logging.getLogger(__name__)

_ISO2_TO_ISO3: dict[str, str] = {
    "GR": "GRC",
    "EL": "GRC",
    "DE": "DEU",
    "FR": "FRA",
    "IT": "ITA",
    "ES": "ESP",
    "LU": "LUX",
    "AL": "ALB",
}


def _to_iso3(code: str) -> str:
    c = str(code).strip().upper()
    if len(c) == 3 and c.isalpha():
        return c
    if len(c) == 2:
        return _ISO2_TO_ISO3.get(c, c)
    return c


def _norm_sector_token(s: str) -> str:
    t = str(s).strip().upper()
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def _parse_float(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    s = str(v).strip().upper()
    if s in ("NA", "N/A", "C", "-", "", "NAN"):
        return float("nan")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return float("nan")


def load_sector_family_rules(yaml_path: Path) -> tuple[dict[str, str], frozenset[str]]:
    """
    Return mapping normalized_token_prefix -> family name 'solid'|'ww'|'res',
    and a frozenset of excluded tokens (exact match after norm on full string).
    """
    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    fams: dict[str, str] = {}
    for family, keys in (raw.get("families") or {}).items():
        for k in keys:
            tok = _norm_sector_token(str(k))
            if tok:
                fams[tok] = str(family)
    excl = frozenset(_norm_sector_token(str(x)) for x in (raw.get("exclude_always") or []))
    return fams, excl


def _family_for_sector(
    sector_norm: str,
    fams: dict[str, str],
    excl: frozenset[str],
) -> str | None:
    if sector_norm in excl:
        return None
    if sector_norm in fams:
        return fams[sector_norm]
    for prefix, fam in fams.items():
        if sector_norm.startswith(prefix):
            return fam
    return None


def _pick_column(cols_lower: dict[str, str], *names: str) -> str | None:
    for n in names:
        if n in cols_lower:
            return cols_lower[n]
    return None


def read_ceip_waste_table(
    xlsx_path: Path,
    *,
    sheet: str | None,
    year: int,
    fams: dict[str, str],
    excl: frozenset[str],
    pollutant_aliases: dict[str, str],
    cam_pollutants_upper: list[str],
) -> pd.DataFrame:
    """
    Return long DataFrame: country_iso3, pollutant (CAMS-like upper), sector_raw,
    E_value (as read from file; assumed Gg unless config adds scaling later).
    """
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"CEIP Excel not found: {xlsx_path}")
    try:
        raw = pd.read_excel(
            xlsx_path,
            sheet_name=sheet if sheet else 0,
            header=0,
            engine="openpyxl",
        )
    except ValueError:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        raw = pd.read_excel(xlsx_path, sheet_name=xl.sheet_names[0], header=0, engine="openpyxl")
    cols = {str(c).lower().strip(): c for c in raw.columns}
    c_country = _pick_column(cols, "country_code", "country", "iso", "iso3", "iso2")
    c_year = _pick_column(cols, "year", "yr")
    c_sector = _pick_column(cols, "sector", "sector_code", "snap", "nfr", "code")
    c_total = _pick_column(cols, "total", "emissions", "value", "emission", "gg")
    c_pol = _pick_column(cols, "pollutant", "compound", "species", "param", "poll")
    if not all([c_country, c_year, c_sector, c_total]):
        raise ValueError(
            f"CEIP sheet missing required columns. Have: {list(raw.columns)}. "
            "Need country, year, sector, total (and optional pollutant)."
        )
    if not c_pol:
        raise ValueError(
            "CEIP sheet must include a pollutant column "
            "(e.g. pollutant, compound, species). "
            f"Columns present: {list(raw.columns)}"
        )
    cam_set = {p.upper().replace(".", "_") for p in cam_pollutants_upper}
    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        try:
            yr = int(float(r[c_year]))
        except (TypeError, ValueError):
            continue
        if yr != int(year):
            continue
        iso3 = _to_iso3(str(r[c_country]))
        if len(iso3) != 3:
            continue
        iso3 = iso3.upper()
        sec_raw = str(r[c_sector])
        sec_n = _norm_sector_token(sec_raw)
        fam = _family_for_sector(sec_n, fams, excl)
        if fam is None:
            continue
        v = _parse_float(r[c_total])
        if not np.isfinite(v) or v < 0:
            continue
        if not c_pol or not str(r[c_pol]).strip():
            continue
        pol = str(r[c_pol]).strip().upper().replace(".", "_")
        pol = pollutant_aliases.get(pol, pol)
        if pol == "PM25":
            pol = "PM2_5"
        if pol not in cam_set:
            continue
        rows.append(
            {
                "country_iso3": iso3,
                "pollutant": pol,
                "sector_raw": sec_raw,
                "family": fam,
                "E_value": v,
            }
        )
    if not rows:
        raise ValueError(
            f"No CEIP rows after filtering (year={year}). Check sector codes and sheet."
        )
    return pd.DataFrame(rows)


def aggregate_family_emissions(long_df: pd.DataFrame) -> pd.DataFrame:
    """Sum E_value per country_iso3, pollutant, family."""
    g = (
        long_df.groupby(["country_iso3", "pollutant", "family"], as_index=False)["E_value"]
        .sum()
        .rename(columns={"E_value": "E_family"})
    )
    pivot = g.pivot_table(
        index=["country_iso3", "pollutant"],
        columns="family",
        values="E_family",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for col in ("solid", "ww", "res"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot


def compute_weights_with_fallbacks(
    pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add E_solid, E_ww, E_res, w_*, fallback_tier, fallback_note.
    Returns (wide table, fallback log rows).
    """
    out = pivot.copy()
    out["E_solid"] = out["solid"].astype(np.float64)
    out["E_ww"] = out["ww"].astype(np.float64)
    out["E_res"] = out["res"].astype(np.float64)
    out["E_sum3"] = out["E_solid"] + out["E_ww"] + out["E_res"]
    out["w_solid"] = np.nan
    out["w_ww"] = np.nan
    out["w_res"] = np.nan
    out["fallback_tier"] = 0
    out["fallback_note"] = ""

    fb_rows: list[dict[str, Any]] = []

    def set_equal(idx: Any) -> None:
        out.loc[idx, "w_solid"] = 1.0 / 3.0
        out.loc[idx, "w_ww"] = 1.0 / 3.0
        out.loc[idx, "w_res"] = 1.0 / 3.0

    # Tier 0: direct normalization
    pos = out["E_sum3"] > 0
    out.loc[pos, "w_solid"] = out.loc[pos, "E_solid"] / out.loc[pos, "E_sum3"]
    out.loc[pos, "w_ww"] = out.loc[pos, "E_ww"] / out.loc[pos, "E_sum3"]
    out.loc[pos, "w_res"] = out.loc[pos, "E_res"] / out.loc[pos, "E_sum3"]

    need = ~pos
    if not need.any():
        return out, pd.DataFrame(fb_rows)

    # Tier 1: same country, other pollutants mean of (w_solid,w_ww,w_res) where E_sum3>0
    for ctry in out.loc[need, "country_iso3"].unique():
        sub_need = need & (out["country_iso3"] == ctry)
        if not sub_need.any():
            continue
        ref = out[(out["country_iso3"] == ctry) & pos]
        if len(ref) == 0:
            continue
        m_s = ref["w_solid"].mean()
        m_w = ref["w_ww"].mean()
        m_r = ref["w_res"].mean()
        ssum = m_s + m_w + m_r
        if np.isfinite(ssum) and ssum > 0:
            idx = sub_need
            out.loc[idx, "w_solid"] = m_s / ssum
            out.loc[idx, "w_ww"] = m_w / ssum
            out.loc[idx, "w_res"] = m_r / ssum
            out.loc[idx, "fallback_tier"] = 1
            out.loc[idx, "fallback_note"] = "country_mean_other_pollutants"
            for i in out.loc[idx].index:
                fb_rows.append(
                    {
                        "country_iso3": out.at[i, "country_iso3"],
                        "pollutant": out.at[i, "pollutant"],
                        "tier": 1,
                        "note": "country_mean_other_pollutants",
                    }
                )

    still_na = out["w_solid"].isna()

    # Tier 2: EU mean over rows with tier0 positive
    ref2 = out[out["E_sum3"] > 0]
    if len(ref2) and still_na.any():
        m_s = ref2["w_solid"].mean()
        m_w = ref2["w_ww"].mean()
        m_r = ref2["w_res"].mean()
        ssum = m_s + m_w + m_r
        if np.isfinite(ssum) and ssum > 0:
            fill = still_na
            out.loc[fill, "w_solid"] = m_s / ssum
            out.loc[fill, "w_ww"] = m_w / ssum
            out.loc[fill, "w_res"] = m_r / ssum
            out.loc[fill, "fallback_tier"] = 2
            out.loc[fill, "fallback_note"] = "eu_mean_valid_countries"
            for i in out.loc[fill].index:
                fb_rows.append(
                    {
                        "country_iso3": out.at[i, "country_iso3"],
                        "pollutant": out.at[i, "pollutant"],
                        "tier": 2,
                        "note": "eu_mean_valid_countries",
                    }
                )

    # Tier 3: equal
    need = out["w_solid"].isna()
    if need.any():
        set_equal(need)
        out.loc[need, "fallback_tier"] = 3
        out.loc[need, "fallback_note"] = "equal_weights"
        for i in out.loc[need].index:
            fb_rows.append(
                {
                    "country_iso3": out.at[i, "country_iso3"],
                    "pollutant": out.at[i, "pollutant"],
                    "tier": 3,
                    "note": "equal_weights",
                }
            )

    return out, pd.DataFrame(fb_rows)


def build_ceip_weight_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry: load Excel + yaml rules, return (long_df, country_pollutant_weights, fallbacks).
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    ceip_cfg = cfg.get("ceip") or {}
    xlsx = resolve_path(root, Path(paths["ceip_xlsx"]))
    ypath = resolve_path(root, Path(paths.get("ceip_sector_codes_yaml", "Waste/j_waste_weights/ceip_sector_codes.yaml")))
    fams, excl = load_sector_family_rules(ypath)
    year = int(paths.get("ceip_year") or ceip_cfg.get("year", 2019))
    sheet = paths.get("ceip_sheet")
    aliases = dict(ceip_cfg.get("pollutant_aliases") or {})
    cam_pol = [str(p).upper() for p in cfg["cams"]["pollutants_nc"]]
    long_df = read_ceip_waste_table(
        xlsx,
        sheet=sheet,
        year=year,
        fams=fams,
        excl=excl,
        pollutant_aliases=aliases,
        cam_pollutants_upper=cam_pol,
    )
    pivot = aggregate_family_emissions(long_df)
    wide, fb = compute_weights_with_fallbacks(pivot)
    logger.info("CEIP family weights: %d country-pollutant rows", len(wide))
    return long_df, wide, fb


def weights_lookup_arrays(
    wide: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (n_iso, n_pol) arrays for w_solid, w_ww, w_res aligned with iso3_list order.

    ``iso3_list[k]`` is ISO3 for country raster value k; index 0 unused.
    Pollutants order matches ``pollutants`` (lower-case CAMS names).
    """
    n_iso = len(iso3_list)
    n_pol = len(pollutants)
    ws = np.full((n_iso, n_pol), 1.0 / 3.0, dtype=np.float64)
    ww = np.full((n_iso, n_pol), 1.0 / 3.0, dtype=np.float64)
    wr = np.full((n_iso, n_pol), 1.0 / 3.0, dtype=np.float64)
    pol_upper = [p.upper().replace(".", "_") for p in pollutants]
    idx_map = {iso: i for i, iso in enumerate(iso3_list) if iso}
    for _, row in wide.iterrows():
        iso = str(row["country_iso3"]).upper()
        pol = str(row["pollutant"]).upper().replace(".", "_")
        if pol not in pol_upper:
            continue
        j = pol_upper.index(pol)
        i = idx_map.get(iso)
        if i is None:
            continue
        ws[i, j] = float(row["w_solid"])
        ww[i, j] = float(row["w_ww"])
        wr[i, j] = float(row["w_res"])
    return ws, ww, wr
