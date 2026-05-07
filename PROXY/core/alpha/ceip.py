"""CEIP reported-emissions reader for the offroad triple-leg shares (1A3c / 1A3ei / 1A3eii).

Wide and long CEIP layouts are both supported. When ``xlsx_path`` is missing, sensible
defaults are returned with a warning so downstream code can continue with uniform shares.
"""
from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ._common import _parse_emission_value, _pick_col
from .aliases import ISO2_TO_ISO3_CEIP, _norm_pol, resolve_country_iso3
from .workbook import (
    _is_semicolon_single_column_table,
    _semicolon_rows_to_standard_columns,
)

logger = logging.getLogger(__name__)


def _offroad_sector_leg(sector_raw) -> int | None:
    s = str(sector_raw).strip().upper().replace(" ", "")
    if "1A3EII" in s or s.endswith("1A3EII"):
        return 2
    if "1A3EI" in s or s.endswith("1A3EI"):
        return 1
    if "1A3C" in s or s.endswith("1A3C"):
        return 0
    alnum = re.sub(r"[^A-Z0-9]", "", s)
    if alnum.endswith("1A3EII"):
        return 2
    if alnum.endswith("1A3EI"):
        return 1
    if alnum.endswith("1A3C"):
        return 0
    return None


def _load_ceip_workbook_tables(
    xlsx_path: Path,
    *,
    sheet: str | int | None,
    ceip_year: int | None,
) -> pd.DataFrame:
    """Load one or all sheets; normalize semicolon-in-one-column layout to wide columns."""
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    names = list(xls.sheet_names)

    if isinstance(sheet, int):
        if sheet < 0 or sheet >= len(names):
            raise ValueError(
                f"CEIP sheet index {sheet} out of range; workbook has {len(names)} sheet(s)."
            )
        sheet_names = [names[sheet]]
    elif isinstance(sheet, str) and sheet.strip():
        if sheet not in names:
            raise ValueError(f"CEIP sheet {sheet!r} not in workbook; available: {names!r}")
        sheet_names = [sheet]
    else:
        sheet_names = names

    chunks: list[pd.DataFrame] = []
    for sn in sheet_names:
        part = pd.read_excel(xlsx_path, sheet_name=sn, header=0, engine="openpyxl")
        if _is_semicolon_single_column_table(part):
            part = _semicolon_rows_to_standard_columns(part)
        chunks.append(part)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def read_ceip_shares(
    xlsx_path: Path,
    *,
    sheet: str | int | None,
    pollutant_aliases: dict[str, str],
    pollutants_wanted: list[str],
    cntr_code_to_iso3: dict[str, str],
    default_triple: tuple[float, float, float],
    ceip_year: int | None = None,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return ``shares[pollutant_key][iso3_upper] = (s_rail, s_pipe, s_nonroad)``.

    Supports wide layout (``E_1A3c / E_1A3ei / E_1A3eii``) or long CEIP-style
    (``COUNTRY, YEAR, SECTOR, POLLUTANT, TOTAL``).

    When ``sheet`` is ``None``, every sheet is loaded (e.g. one sheet per year). In long
    format the triple-leg shares are the mean of per-calendar-year shares (each year
    normalized first, then averaged across years).
    """
    if not xlsx_path.is_file():
        logger.warning("CEIP workbook missing (%s); using defaults only.", xlsx_path)
        iso = str(cntr_code_to_iso3.get("EL", "GRC")).upper()
        d_rail, d_pipe, d_nr = default_triple
        return {p: {iso: (d_rail, d_pipe, d_nr)} for p in pollutants_wanted}

    raw = _load_ceip_workbook_tables(xlsx_path, sheet=sheet, ceip_year=ceip_year)

    c_country = _pick_col(raw, ("country", "iso3", "country_code", "cntr_code", "iso", "c_country"))
    c_pol = _pick_col(raw, ("pollutant", "species", "compound"))
    c1 = _pick_col(raw, ("e_1a3c", "1a3c", "eea_1a3c"))
    c2 = _pick_col(raw, ("e_1a3ei", "1a3ei", "eea_1a3ei"))
    c3 = _pick_col(raw, ("e_1a3eii", "1a3eii", "eea_1a3eii"))

    cntr_map = dict(cntr_code_to_iso3)
    for k, v in list(ISO2_TO_ISO3_CEIP.items()):
        cntr_map.setdefault(k, v)

    wide_ok = bool(c_country and c_pol and c1 and c2 and c3)

    if wide_ok:
        return _read_ceip_wide(
            raw,
            c_country=c_country,
            c_pol=c_pol,
            c1=c1,
            c2=c2,
            c3=c3,
            pollutant_aliases=pollutant_aliases,
            pollutants_wanted=pollutants_wanted,
            cntr_map=cntr_map,
            default_triple=default_triple,
        )

    c_year = _pick_col(raw, ("year", "yr"))
    c_sector = _pick_col(raw, ("sector", "sector_code", "snap", "nfr", "activity"))
    c_total = _pick_col(raw, ("total", "emissions", "value", "emission", "gg", "kt"))
    if not all([c_country, c_pol, c_sector, c_total]):
        raise ValueError(
            "CEIP Offroad: either wide columns (country, pollutant, E_1A3c, E_1A3ei, E_1A3eii) "
            "or long columns (country, pollutant, sector, total). "
            f"Got columns: {list(raw.columns)}"
        )

    return _read_ceip_long(
        raw,
        c_country=c_country,
        c_pol=c_pol,
        c_sector=c_sector,
        c_total=c_total,
        c_year=c_year,
        pollutant_aliases=pollutant_aliases,
        pollutants_wanted=pollutants_wanted,
        cntr_map=cntr_map,
        default_triple=default_triple,
        ceip_year=ceip_year,
    )


def _read_ceip_wide(
    raw: pd.DataFrame,
    *,
    c_country: str,
    c_pol: str,
    c1: str,
    c2: str,
    c3: str,
    pollutant_aliases: dict[str, str],
    pollutants_wanted: list[str],
    cntr_map: dict[str, str],
    default_triple: tuple[float, float, float],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    wanted = {_norm_pol(p) for p in pollutants_wanted}
    out: dict[str, dict[str, tuple[float, float, float]]] = {
        _norm_pol(p): {} for p in pollutants_wanted
    }

    for _, row in raw.iterrows():
        cc = resolve_country_iso3(row[c_country], cntr_map)
        if not cc or len(cc) != 3:
            continue
        pol_raw = row[c_pol]
        su = str(pol_raw).strip().upper().replace(".", "_")
        pol = pollutant_aliases.get(su, pollutant_aliases.get(str(pol_raw).strip().upper(), str(pol_raw)))
        polk = _norm_pol(pol)
        if polk not in wanted:
            continue
        try:
            e1 = float(row[c1])
            e2 = float(row[c2])
            e3 = float(row[c3])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(e1) and np.isfinite(e2) and np.isfinite(e3)):
            continue
        et = e1 + e2 + e3
        if et <= 0:
            continue
        out.setdefault(polk, {})[cc] = (e1 / et, e2 / et, e3 / et)

    _fill_ceip_defaults(out, pollutants_wanted, cntr_map, default_triple)
    return out


def _read_ceip_long(
    raw: pd.DataFrame,
    *,
    c_country: str,
    c_pol: str,
    c_sector: str,
    c_total: str,
    c_year: str | None,
    pollutant_aliases: dict[str, str],
    pollutants_wanted: list[str],
    cntr_map: dict[str, str],
    default_triple: tuple[float, float, float],
    ceip_year: int | None,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    _ = ceip_year
    wanted = {_norm_pol(p) for p in pollutants_wanted}
    out: dict[str, dict[str, tuple[float, float, float]]] = {
        _norm_pol(p): {} for p in pollutants_wanted
    }

    df = raw.copy()
    if c_year is not None and c_year in df.columns:
        yn = pd.to_numeric(df[c_year], errors="coerce")
        df = df.assign(_ceip_yr_bucket=yn)
    else:
        df = df.assign(_ceip_yr_bucket=0)

    triple_lists: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)

    for _y_val, g in df.groupby("_ceip_yr_bucket", dropna=True):
        acc: dict[tuple[str, str], list[float]] = {}

        for _, row in g.iterrows():
            cc = resolve_country_iso3(row[c_country], cntr_map)
            if not cc or len(cc) != 3:
                continue

            leg = _offroad_sector_leg(row[c_sector])
            if leg is None:
                continue

            pol_raw = row[c_pol]
            su = str(pol_raw).strip().upper().replace(".", "_")
            pol = pollutant_aliases.get(su, pollutant_aliases.get(str(pol_raw).strip().upper(), str(pol_raw)))
            polk = _norm_pol(pol)
            if polk not in wanted:
                continue

            val = _parse_emission_value(row[c_total])
            if not np.isfinite(val) or val < 0:
                continue

            key = (cc, polk)
            if key not in acc:
                acc[key] = [0.0, 0.0, 0.0]
            acc[key][leg] += float(val)

        for (iso3, polk), trip in acc.items():
            et = trip[0] + trip[1] + trip[2]
            if et <= 0:
                continue
            triple_lists[(iso3, polk)].append(
                (trip[0] / et, trip[1] / et, trip[2] / et),
            )

    for (iso3, polk), trips in triple_lists.items():
        arr = np.asarray(trips, dtype=np.float64)
        m = arr.mean(axis=0)
        s = float(m.sum())
        if s > 0:
            m = m / s
        out.setdefault(polk, {})[iso3] = (float(m[0]), float(m[1]), float(m[2]))

    logger.info(
        "CEIP Offroad (long format, mean of annual shares): loaded %d country x pollutant keys.",
        sum(len(v) for v in out.values()),
    )
    _fill_ceip_defaults(out, pollutants_wanted, cntr_map, default_triple)
    return out


def _fill_ceip_defaults(
    out: dict[str, dict[str, tuple[float, float, float]]],
    pollutants_wanted: list[str],
    cntr_code_to_iso3: dict[str, str],
    default_triple: tuple[float, float, float],
) -> None:
    d_rail, d_pipe, d_nr = default_triple
    iso_fb = str(cntr_code_to_iso3.get("EL", cntr_code_to_iso3.get("GR", "GRC"))).upper()
    for p in pollutants_wanted:
        pk = _norm_pol(p)
        if pk not in out:
            out[pk] = {}
        if not out[pk]:
            out[pk] = {iso_fb: (d_rail, d_pipe, d_nr)}
        else:
            for iso3, trip in list(out[pk].items()):
                if sum(trip) <= 0:
                    del out[pk][iso3]
            if not out[pk]:
                out[pk] = {iso_fb: (d_rail, d_pipe, d_nr)}
