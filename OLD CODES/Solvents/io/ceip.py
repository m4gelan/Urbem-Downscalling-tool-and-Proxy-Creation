"""CEIP national totals (2.D.3) -> alpha_{country, subsector, pollutant}."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import resolve_path as project_resolve


def _subsector_for_nfr(sector_norm: str, tok2sub: dict[str, str], subsectors: set[str]) -> str | None:
    """Match longest NFR token first (same idea as D_Fugitive group mapping)."""
    sn = str(sector_norm).strip().upper()
    if not sn:
        return None
    for k in sorted(tok2sub.keys(), key=lambda x: -len(x)):
        if sn == k or sn.startswith(k):
            sub = tok2sub[k]
            if sub in subsectors:
                return sub
    return None


def load_nfr_to_subsector_map(root: Path, cfg: dict[str, Any]) -> dict[str, str]:
    """YAML: ``subsectors.<key>.ceip_sectors: [NFR, ...]`` -> token -> subsector key."""
    import yaml

    p = project_resolve(root, Path(cfg["paths"]["ceip_subsector_map_yaml"]))
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    subs = raw.get("subsectors") or {}
    tok2sub: dict[str, str] = {}
    for sub_key, spec in subs.items():
        for sec in (spec or {}).get("ceip_sectors", []) or []:
            t = _norm_sector_token(str(sec))
            if t:
                tok2sub[t] = str(sub_key)
    return tok2sub


def read_reported_emissions_subsector_long(
    xlsx_path: Path,
    *,
    tok2sub: dict[str, str],
    pollutant_aliases: dict[str, str],
    subsectors: list[str],
    years_filter: list[int] | None,
    cntr_code_to_iso3: dict[str, str],
) -> pd.DataFrame:
    """
    ``Reported_Emissions_*.xlsx`` (``PROXY.core.alpha`` layout): map NFR rows to
    2.D.3 subsector keys, return long **mean** annual mass ``E`` per
    (country, pollutant, subsector) — mean across calendar years, same as
    ``PROXY.core.ceip.reported_group_alpha.read_reported_emissions_fugitive_long``.
    """
    from PROXY.core.alpha import read_alpha_workbook
    from PROXY.core.alpha.reported import resolve_iso3_reported

    sub_s = {str(s) for s in subsectors}
    raw = read_alpha_workbook(xlsx_path)
    if raw.empty:
        raise ValueError(f"CEIP reported workbook is empty: {xlsx_path}")
    if years_filter is not None:
        years_set = {int(x) for x in years_filter}
        raw = raw.loc[raw["YEAR"].astype(int).isin(years_set)].copy()
        if raw.empty:
            raise ValueError(
                f"No rows after year filter {sorted(years_set)} in {xlsx_path}"
            )
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
        sub = _subsector_for_nfr(sn, tok2sub, sub_s)
        if sub is None:
            continue
        v = _parse_value(r.get("TOTAL_VALUE"))
        if not np.isfinite(v) or v < 0:
            continue
        su = str(r.get("POLLUTANT", "")).strip().upper().replace(".", "_")
        u0 = pollutant_aliases.get(su, su) if su else "TOTAL"
        u = str(u0).strip().upper().replace(".", "_")
        if u == "PM25":
            u = "PM2_5"
        if not u or u in ("", "NAN"):
            u = "TOTAL"
        rows.append(
            {
                "country_iso3": iso3.upper(),
                "pollutant": u,
                "subsector": sub,
                "E": float(v),
                "YEAR": int(r["YEAR"]),
            }
        )
    if not rows:
        raise ValueError(
            f"No reported-emission rows matched ceip_subsector_map in {xlsx_path}. "
            "Check NFR codes vs PROXY/config/solvents/ceip_subsectors.yaml."
        )
    dfp = pd.DataFrame(rows)
    gsum = dfp.groupby(
        ["country_iso3", "YEAR", "pollutant", "subsector"],
        as_index=False,
    )["E"].sum()
    m = gsum.groupby(
        ["country_iso3", "pollutant", "subsector"],
        as_index=False,
    )["E"].mean()
    return m

# CEIP / EU country codes often ISO2; CAMS uses ISO3.
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


def _parse_value(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    s = str(v).strip().upper()
    if s in ("NA", "N/A", "C", "-", ""):
        return float("nan")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return float("nan")


def load_sector_map(root: Path, cfg: dict[str, Any]) -> dict[str, str]:
    p = project_resolve(root, Path(cfg["paths"]["ceip_sector_map"]))
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(k).upper(): str(v) for k, v in (data.get("aliases") or {}).items()}


def _load_alpha_reported_eu27(
    root: Path,
    cfg: dict[str, Any],
    *,
    target_iso3: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    ``Reported_Emissions_*.xlsx`` + ``ceip_subsector_map_yaml``: mean-over-years
    masses, then alpha per (country, pollutant) over subsectors (same pattern as D_Fugitive).
    """
    paths = cfg["paths"]
    wb = project_resolve(root, Path(paths["ceip_workbook"]))
    subsectors: list[str] = list(cfg["subsectors"])
    pollutants_cfg = [str(p) for p in cfg["pollutants"]]
    tok2sub = load_nfr_to_subsector_map(root, cfg)
    yf = cfg.get("ceip_years")
    years_filter: list[int] | None
    if yf is None or (isinstance(yf, list) and len(yf) == 0):
        years_filter = None
    else:
        years_filter = [int(x) for x in (yf if isinstance(yf, (list, tuple)) else [yf])]
    aliases = dict(cfg.get("ceip_pollutant_aliases") or {})
    cntr = dict(cfg.get("cntr_code_to_iso3") or {})

    m = read_reported_emissions_subsector_long(
        wb,
        tok2sub=tok2sub,
        pollutant_aliases=aliases,
        subsectors=subsectors,
        years_filter=years_filter,
        cntr_code_to_iso3=cntr,
    )
    m = m.loc[m["country_iso3"].str.upper() == target_iso3.upper()].copy()
    m = m.loc[m["pollutant"].isin([p.upper() for p in pollutants_cfg])].copy()
    if m.empty:
        raise ValueError(
            f"No reported-emission rows for {target_iso3!r} after filters "
            f"(pollutants={pollutants_cfg})."
        )
    pol_map = {str(p).upper(): str(p) for p in pollutants_cfg}
    m["pollutant"] = m["pollutant"].map(lambda x: pol_map.get(str(x).upper(), x))

    pieces: list[pd.DataFrame] = []
    for (_, pol), g in m.groupby(["country_iso3", "pollutant"], sort=False):
        g = g.copy()
        s = float(g["E"].sum())
        g["alpha"] = np.where(np.isfinite(s) & (s > 0), g["E"] / s, np.nan)
        g["E_gg"] = g["E"]
        pieces.append(g)
    agg = pd.concat(pieces, ignore_index=True)
    meta = {
        "source": "reported_emissions_eu27",
        "path": str(wb),
        "ceip_years": years_filter,
        "ceip_subsector_map": str(
            project_resolve(root, Path(paths["ceip_subsector_map_yaml"]))
        ),
    }
    return agg, meta


def load_ceip_alpha_table(
    root: Path,
    cfg: dict[str, Any],
    *,
    target_iso3: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return long DataFrame columns: country_iso3, pollutant, subsector, E_kg (Gg converted),
    alpha; and meta dict.
    """
    paths = cfg["paths"]
    subsectors: list[str] = list(cfg["subsectors"])
    mode = str(cfg.get("ceip_alpha_mode", "reported_eu27")).lower()
    wb = project_resolve(root, Path(paths.get("ceip_workbook", "")))
    if mode in ("reported_eu27", "reported", "eu27") and wb.is_file():
        return _load_alpha_reported_eu27(root, cfg, target_iso3=target_iso3)

    xlsx = project_resolve(root, Path(paths.get("ceip_xlsx", "")))
    pollutants_cfg: list[str] = list(cfg["pollutants"])
    sector_map = load_sector_map(root, cfg)
    year_filter = int(cfg.get("ceip_year", 2020))

    if not xlsx.is_file():
        fb = project_resolve(root, Path(paths["fallback_alpha_json"]))
        with fb.open(encoding="utf-8") as f:
            fb_data = json.load(f)
        rows = []
        iso = target_iso3.upper()
        block = fb_data.get(iso) or fb_data.get("GRC", {})
        for pol, subdict in block.items():
            if pol not in pollutants_cfg:
                continue
            for s in subsectors:
                a = float(subdict.get(s, 0.0))
                rows.append(
                    {
                        "country_iso3": iso,
                        "pollutant": pol,
                        "subsector": s,
                        "E_gg": float("nan"),
                        "alpha": a,
                    }
                )
        df = pd.DataFrame(rows)
        meta = {"source": "fallback_json", "path": str(fb)}
        return df, meta

    sheet = paths.get("ceip_sheet", "Sheet1")
    try:
        raw = pd.read_excel(
            xlsx,
            sheet_name=sheet,
            header=0,
            engine="openpyxl",
        )
    except ValueError as exc:
        if "not found" not in str(exc).lower():
            raise
        xl = pd.ExcelFile(xlsx, engine="openpyxl")
        if not xl.sheet_names:
            raise ValueError(f"No sheets in CEIP workbook: {xlsx}") from exc
        sheet = xl.sheet_names[0]
        raw = pd.read_excel(
            xlsx,
            sheet_name=sheet,
            header=0,
            engine="openpyxl",
        )
    cols = {c.lower().strip(): c for c in raw.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_country = pick("country_code", "country", "iso")
    c_year = pick("year")
    c_sector = pick("sector", "sector_code", "snap")
    c_total = pick("total", "emissions", "value")
    c_pol = pick("pollutant", "compound", "species", "param")

    if not all([c_country, c_year, c_sector, c_total]):
        raise ValueError(
            f"CEIP sheet missing required columns. Have: {list(raw.columns)}. "
            "Need country, year, sector, total (and optional pollutant)."
        )

    rows_out: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        try:
            yr = int(float(r[c_year]))
        except (TypeError, ValueError):
            continue
        if yr != year_filter:
            continue
        iso3 = _to_iso3(str(r[c_country]))
        if iso3.upper() != target_iso3.upper():
            continue
        tok = _norm_sector_token(str(r[c_sector]))
        sub = sector_map.get(tok)
        if sub is None:
            for k, v in sector_map.items():
                if tok.startswith(k) or k in tok:
                    sub = v
                    break
        if sub is None or sub not in subsectors:
            continue
        if c_pol and str(r[c_pol]).strip():
            pol = str(r[c_pol]).strip().upper().replace(".", "_")
            if pol == "PM25":
                pol = "PM2_5"
        else:
            pol = "NMVOC"
        if pol not in pollutants_cfg:
            continue
        v_gg = _parse_value(r[c_total])
        if not np.isfinite(v_gg):
            continue
        rows_out.append(
            {
                "country_iso3": iso3.upper(),
                "pollutant": pol,
                "subsector": sub,
                "E_gg": v_gg,
            }
        )

    if not rows_out:
        raise ValueError(
            f"No CEIP rows for {target_iso3!r} year={year_filter}. Check sector map and sheet."
        )

    df = pd.DataFrame(rows_out)
    agg = df.groupby(["country_iso3", "pollutant", "subsector"], as_index=False)["E_gg"].sum()
    pieces: list[pd.DataFrame] = []
    for (_, _), g in agg.groupby(["country_iso3", "pollutant"], sort=False):
        g = g.copy()
        s = float(g["E_gg"].sum())
        g["alpha"] = np.where(np.isfinite(s) & (s > 0), g["E_gg"] / s, np.nan)
        pieces.append(g)
    agg = pd.concat(pieces, ignore_index=True)
    meta = {"source": "ceip_xlsx", "path": str(xlsx), "year": year_filter, "sheet": sheet}
    return agg, meta


def validate_alpha(agg: pd.DataFrame, subsectors: list[str], tol: float = 1e-4) -> list[str]:
    """
    Require each (country, pollutant) group in ``agg`` to have alpha summing to ~1 over
    the subsectors present in that group. A pollutant may use only a subset of subsectors
    (zeros elsewhere are applied after ``alpha_matrix`` + ``finalize_alpha_matrix``).
    """
    errs: list[str] = []
    for (_, pol), g in agg.groupby(["country_iso3", "pollutant"]):
        s = g["alpha"].sum(skipna=True)
        if not np.isfinite(s) or abs(s - 1.0) > tol:
            errs.append(f"alpha sum for {pol!r} = {s} (expected 1)")
    return errs
