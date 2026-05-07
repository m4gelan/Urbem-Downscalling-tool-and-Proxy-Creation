"""CEIP/NRD fugitive table -> group totals and alpha_{country,group,pollutant}."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config_utils import resolve

logger = logging.getLogger(__name__)

_GROUP_IDS = ("G1", "G2", "G3", "G4")
_ISO2_TO_ISO3: dict[str, str] = {"EL": "GRC", "GR": "GRC"}


def _norm_sector(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).strip().upper())


def _to_iso3(code: str) -> str:
    c = str(code).strip().upper()
    if len(c) == 3 and c.isalpha():
        return c
    if len(c) == 2:
        return _ISO2_TO_ISO3.get(c, c)
    return c


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


def load_group_mapping(yaml_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return (token_upper -> G1|G2|G3|G4, raw groups dict)."""
    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    groups = raw.get("groups") or {}
    tok2g: dict[str, str] = {}
    for gid, spec in groups.items():
        if gid not in _GROUP_IDS:
            continue
        for sec in spec.get("ceip_sectors", []) or []:
            t = _norm_sector(str(sec))
            if t:
                tok2g[t] = gid
    return tok2g, groups


def _pick_col(cols: dict[str, str], *names: str) -> str | None:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def read_ceip_long(
    xlsx_path: Path,
    *,
    sheet: str | None,
    year: int,
    tok2g: dict[str, str],
    pollutant_aliases: dict[str, str],
) -> pd.DataFrame:
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"CEIP fugitive workbook not found: {xlsx_path}")
    try:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet if sheet else 0, header=0, engine="openpyxl")
    except ValueError:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        raw = pd.read_excel(xlsx_path, sheet_name=xl.sheet_names[0], header=0, engine="openpyxl")
    cols = {str(c).lower().strip(): c for c in raw.columns}
    c_country = _pick_col(cols, "country_code", "country", "iso", "iso3", "iso2")
    c_year = _pick_col(cols, "year", "yr")
    c_sector = _pick_col(cols, "sector", "sector_code", "snap", "nfr", "code", "nace")
    c_total = _pick_col(cols, "total", "emissions", "value", "emission", "gg", "kt")
    c_pol = _pick_col(cols, "pollutant", "compound", "species", "param", "poll")
    if not all([c_country, c_year, c_sector, c_total]):
        raise ValueError(
            f"CEIP fugitive sheet missing columns. Have: {list(raw.columns)}. "
            "Need country, year, sector, total."
        )
    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        try:
            yr = int(float(r[c_year]))
        except (TypeError, ValueError):
            continue
        if yr != int(year):
            continue
        iso3 = _to_iso3(str(r[c_country]))
        if len(iso3) != 3 or not iso3.isalpha():
            continue
        iso3 = iso3.upper()
        tok = _norm_sector(str(r[c_sector]))
        gid = None
        for k, g in tok2g.items():
            if tok == k or tok.startswith(k):
                gid = g
                break
        if gid is None:
            continue
        v = _parse_float(r[c_total])
        if not np.isfinite(v) or v < 0:
            continue
        if c_pol and str(r[c_pol]).strip():
            pol = str(r[c_pol]).strip().upper().replace(".", "_")
            pol = pollutant_aliases.get(pol, pol)
            if pol == "PM25":
                pol = "PM2_5"
        else:
            pol = "TOTAL"
        rows.append({"country_iso3": iso3, "pollutant": pol, "group": gid, "E": v})
    if not rows:
        raise ValueError(f"No CEIP fugitive rows after filter (year={year}).")
    return pd.DataFrame(rows)


def build_alpha_tensor(
    long_df: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Return ``(alpha, fallback_per_pol, wide_debug)``.

    ``alpha`` shape ``(n_iso, 4, n_pol)`` aligned with ``iso3_list`` row index and
    pollutant order (lower-case CAMS names in ``pollutants``).

    Per country and pollutant: if group totals sum to zero, fall back to CEIP
    ``TOTAL`` rows by group for that country; if still zero, fall back to the
    country’s group totals summed across **all** pollutants (excluding the
    synthetic ``TOTAL`` pollutant label); if still zero, use uniform 1/4.

    ``fallback_code`` in the wide table is per (country, pollutant): 0 none,
    1 used TOTAL rows, 2 used all-pollutant sum, 3 uniform.
    """
    n_iso = len(iso3_list)
    n_pol = len(pollutants)
    gid_to_i = {g: i for i, g in enumerate(_GROUP_IDS)}
    iso_to_row = {iso3_list[i].upper(): i for i in range(n_iso) if iso3_list[i]}

    agg = long_df.groupby(["country_iso3", "pollutant", "group"], as_index=False)["E"].sum()
    tot = long_df[long_df["pollutant"] == "TOTAL"].groupby(["country_iso3", "group"], as_index=False)["E"].sum()
    agg_nop = agg[agg["pollutant"] != "TOTAL"].groupby(["country_iso3", "group"], as_index=False)["E"].sum()

    alpha = np.full((n_iso, 4, n_pol), 0.25, dtype=np.float64)
    fallback = np.zeros((n_iso, n_pol), dtype=np.int32)

    pol_norm = agg["pollutant"].str.upper().str.replace(".", "_", regex=False)

    for j, pl in enumerate([p.upper().replace(".", "_") for p in pollutants]):
        sub = agg.loc[pol_norm == pl]
        for ri in range(n_iso):
            iso = str(iso3_list[ri]).strip().upper()
            if not iso:
                continue
            vec = np.zeros(4, dtype=np.float64)
            for _, row in sub[sub["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec[gi] += float(row["E"])
            if float(vec.sum()) > 0:
                alpha[ri, :, j] = vec / vec.sum()
                continue
            vec2 = np.zeros(4, dtype=np.float64)
            for _, row in tot[tot["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec2[gi] += float(row["E"])
            if float(vec2.sum()) > 0:
                alpha[ri, :, j] = vec2 / vec2.sum()
                fallback[ri, j] = 1
                continue
            vec3 = np.zeros(4, dtype=np.float64)
            for _, row in agg_nop[agg_nop["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec3[gi] += float(row["E"])
            if float(vec3.sum()) > 0:
                alpha[ri, :, j] = vec3 / vec3.sum()
                fallback[ri, j] = 2
            else:
                alpha[ri, :, j] = 0.25
                fallback[ri, j] = 3

    wide_rows: list[dict[str, Any]] = []
    for j, pl in enumerate(pollutants):
        for ri in range(n_iso):
            if not iso3_list[ri]:
                continue
            wide_rows.append(
                {
                    "country_iso3": iso3_list[ri],
                    "pollutant": pl.lower(),
                    "alpha_G1": alpha[ri, 0, j],
                    "alpha_G2": alpha[ri, 1, j],
                    "alpha_G3": alpha[ri, 2, j],
                    "alpha_G4": alpha[ri, 3, j],
                    "fallback_code": int(fallback[ri, j]),
                }
            )
    wide = pd.DataFrame(wide_rows)
    return alpha, fallback, wide


def load_ceip_and_alpha(cfg: dict[str, Any], iso3_list: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    root: Path = cfg["_project_root"]
    paths = cfg["paths"]
    ypath = resolve(root, Path(paths["ceip_groups_yaml"]))
    tok2g, _ = load_group_mapping(ypath)
    xlsx = resolve(root, Path(paths["ceip_xlsx"]))
    year = int(paths.get("ceip_year", 2019))
    sheet = paths.get("ceip_sheet")
    aliases = dict((cfg.get("ceip_pollutant_aliases") or {}))
    long_df = read_ceip_long(xlsx, sheet=sheet, year=year, tok2g=tok2g, pollutant_aliases=aliases)
    pollutants = [str(p) for p in cfg["pollutants"]]
    alpha, fb, wide = build_alpha_tensor(long_df, iso3_list, pollutants)
    logger.info("CEIP fugitive alpha tensor: iso=%d groups=4 pollutants=%d", len(iso3_list), len(pollutants))
    return alpha, fb, wide
