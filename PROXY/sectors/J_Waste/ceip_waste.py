"""
CEIP / reported emissions -> family weights (w_solid, w_ww, w_res) per country and pollutant.

When ``ceip_alpha_mode`` is ``reported_eu27``, uses ``PROXY.core.alpha`` workbook layout
(mean over years). Otherwise legacy single-year CEIP Excel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY.core.alpha.fallback import AlphaSource, format_provenance, resolve_alpha
from PROXY.core.alpha.reported import (
    normalize_sector_token,
    parse_float_or_nan,
    resolve_iso3_reported,
    short_country,
    to_iso3,
)
from PROXY.core.ceip import default_ceip_profile_relpath, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import resolve_path as project_resolve

logger = logging.getLogger(__name__)


_J_WASTE_SUBSECTORS: tuple[str, ...] = ("solid", "ww", "res")
_ISO3_TO_SHORT_JW: dict[str, str] = {"GRC": "EL"}


def _short_country_jw(iso3: str) -> str:
    return short_country(iso3, _ISO3_TO_SHORT_JW)


def _apply_j_waste_fallback_yaml(
    out: pd.DataFrame,
    *,
    focus_country_iso3: str | None = None,
) -> None:
    """Phase 2.3/2.4: route ``fallback_tier == 3`` rows (uniform 1/3) through the
    shared :func:`core.alpha.fallback.resolve_alpha`. YAML overrides, when present,
    replace the equal-weight values in place; when the YAML is empty the resolver
    returns 1/3 for each leg and behaviour is unchanged.
    """
    if "fallback_tier" not in out.columns:
        return
    focus_u = str(focus_country_iso3 or "").strip().upper()
    uniform_idx = out.index[out["fallback_tier"] == 3]
    for i in uniform_idx:
        iso3 = str(out.at[i, "country_iso3"])
        pol = str(out.at[i, "pollutant"])
        country_short = _short_country_jw(iso3)
        iso3u = str(iso3).strip().upper()
        res = resolve_alpha(
            sector="J_Waste",
            country=country_short,
            pollutant=pol,
            subsectors=list(_J_WASTE_SUBSECTORS),
        )
        out.at[i, "w_solid"] = float(res.values.get("solid", out.at[i, "w_solid"]))
        out.at[i, "w_ww"] = float(res.values.get("ww", out.at[i, "w_ww"]))
        out.at[i, "w_res"] = float(res.values.get("res", out.at[i, "w_res"]))
        if any(s is not AlphaSource.UNIFORM_FALLBACK for s in res.source.values()):
            msg = "[alpha] sector=J_Waste country=%s pollutant=%s %s"
            args_t = (country_short, pol, format_provenance(res))
            if not focus_u or iso3u == focus_u:
                logger.info(msg, *args_t)
            else:
                logger.debug(msg, *args_t)

def _to_iso3(code: str) -> str:
    return to_iso3(code, {"GR": "GRC", "EL": "GRC"})


def _norm_sector_token(s: str) -> str:
    return normalize_sector_token(s)


def _parse_float(v: Any) -> float:
    return parse_float_or_nan(v)


def load_sector_family_rules(yaml_path: Path) -> tuple[dict[str, str], frozenset[str]]:
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


def read_reported_waste_families_long(
    xlsx_path: Path,
    *,
    fams: dict[str, str],
    excl: frozenset[str],
    pollutant_aliases: dict[str, str],
    cam_pollutants_upper: list[str],
    years_filter: list[int] | None,
    cntr_code_to_iso3: dict[str, str],
) -> pd.DataFrame:
    from PROXY.core.alpha import read_alpha_workbook

    cam_set = {p.upper().replace(".", "_") for p in cam_pollutants_upper}
    raw = read_alpha_workbook(xlsx_path)
    if raw.empty:
        raise ValueError(f"Reported-emissions workbook is empty: {xlsx_path}")
    if years_filter is not None:
        years_set = {int(x) for x in years_filter}
        raw = raw.loc[raw["YEAR"].astype(int).isin(years_set)].copy()
        if raw.empty:
            raise ValueError(f"No rows after year filter {sorted(years_set)} in {xlsx_path}")
    if "SECTOR_NORM" not in raw.columns and "SECTOR" in raw.columns:
        raw["SECTOR_NORM"] = raw["SECTOR"].map(_norm_sector_token)
    if "SECTOR_NORM" not in raw.columns:
        raise ValueError("Workbook has no SECTOR/SECTOR_NORM after parse.")

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        cc_raw = r.get("COUNTRY")
        iso3 = resolve_iso3_reported(str(cc_raw), cntr_code_to_iso3)
        if not iso3 or len(iso3) != 3:
            continue
        sn = str(r.get("SECTOR_NORM", _norm_sector_token(r.get("SECTOR", "")))).strip()
        fam = _family_for_sector(sn, fams, excl)
        if fam is None:
            continue
        v = _parse_float(r.get("TOTAL_VALUE"))
        if not np.isfinite(v) or v < 0:
            continue
        su = str(r.get("POLLUTANT", "")).strip().upper().replace(".", "_")
        u0 = pollutant_aliases.get(su, su) if su else "TOTAL"
        pol = str(u0).strip().upper().replace(".", "_")
        if pol == "PM25":
            pol = "PM2_5"
        if not pol or pol in ("", "NAN"):
            continue
        if pol not in cam_set:
            continue
        rows.append(
            {
                "country_iso3": iso3.upper(),
                "pollutant": pol,
                "family": fam,
                "E_value": float(v),
                "YEAR": int(r["YEAR"]),
            }
        )
    if not rows:
        raise ValueError(
            f"No reported rows matched waste ceip_families in {xlsx_path}. "
            "Check NFR/SNAP tokens vs PROXY/config/ceip/profiles/waste_families.yaml."
        )
    dfp = pd.DataFrame(rows)
    g = dfp.groupby(
        ["country_iso3", "YEAR", "pollutant", "family"],
        as_index=False,
    )["E_value"].sum()
    m = g.groupby(
        ["country_iso3", "pollutant", "family"],
        as_index=False,
    )["E_value"].mean()
    return m


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
            "CEIP sheet must include a pollutant column. "
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
    *,
    focus_country_iso3: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    pos = out["E_sum3"] > 0
    out.loc[pos, "w_solid"] = out.loc[pos, "E_solid"] / out.loc[pos, "E_sum3"]
    out.loc[pos, "w_ww"] = out.loc[pos, "E_ww"] / out.loc[pos, "E_sum3"]
    out.loc[pos, "w_res"] = out.loc[pos, "E_res"] / out.loc[pos, "E_sum3"]

    need = ~pos
    if not need.any():
        return out, pd.DataFrame(fb_rows)

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

    _apply_j_waste_fallback_yaml(out, focus_country_iso3=focus_country_iso3)

    return out, pd.DataFrame(fb_rows)


def build_ceip_weight_tables(
    cfg: dict[str, Any],
    *,
    focus_country_iso3: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root: Path = cfg["_project_root"]
    paths = cfg["paths"]
    ceip_cfg = cfg.get("ceip") or {}
    mode = str(ceip_cfg.get("ceip_alpha_mode", "reported_eu27")).lower()
    _fam_rel = str(
        paths.get(
            "ceip_families_yaml",
            default_ceip_profile_relpath(root, "J_Waste", "families_yaml"),
        )
    )
    ypath = project_resolve(root, Path(remap_legacy_ceip_relpath(_fam_rel)))
    fams, excl = load_sector_family_rules(ypath)
    cam_pol = [str(p).upper() for p in cfg["cams"]["pollutants_nc"]]
    aliases = dict(ceip_cfg.get("pollutant_aliases") or {})
    yf = ceip_cfg.get("ceip_years")
    years_filter: list[int] | None
    if yf is None or (isinstance(yf, list) and len(yf) == 0):
        years_filter = None
    else:
        years_filter = [int(x) for x in (yf if isinstance(yf, (list, tuple)) else [yf])]
    cntr = dict(ceip_cfg.get("cntr_code_to_iso3") or {})

    wb = project_resolve(root, Path(paths.get("ceip_workbook", "")))
    if mode in ("reported_eu27", "reported", "eu27") and wb.is_file():
        long_df = read_reported_waste_families_long(
            wb,
            fams=fams,
            excl=excl,
            pollutant_aliases=aliases,
            cam_pollutants_upper=cam_pol,
            years_filter=years_filter,
            cntr_code_to_iso3=cntr,
        )
    else:
        xlsx = project_resolve(root, Path(paths.get("ceip_xlsx", "")))
        if not xlsx.is_file():
            raise FileNotFoundError(
                f"No CEIP workbook (reported) and no legacy ceip_xlsx: {xlsx} / {wb}"
            )
        year = int(paths.get("ceip_year") or ceip_cfg.get("year", 2019))
        sheet = paths.get("ceip_sheet")
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
    wide, fb = compute_weights_with_fallbacks(pivot, focus_country_iso3=focus_country_iso3)
    logger.info("CEIP family weights: %d country-pollutant rows", len(wide))
    return long_df, wide, fb


def renormalize_solid_ww_res_weights(
    ws: np.ndarray, ww: np.ndarray, wr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per (country row, pollutant) column, enforce finite weights and sum=1.
    Replaces all-zero, NaN, or negative sums with 1/3, 1/3, 1/3 and renormalizes.
    """
    t = ws + ww + wr
    bad = ~np.isfinite(t) | (t <= 0)
    u = 1.0 / 3.0
    wsa = np.where(bad, u, ws)
    wwa = np.where(bad, u, ww)
    wra = np.where(bad, u, wr)
    t2 = wsa + wwa + wra
    return wsa / t2, wwa / t2, wra / t2


def weights_lookup_arrays(
    wide: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return renormalize_solid_ww_res_weights(ws, ww, wr)
