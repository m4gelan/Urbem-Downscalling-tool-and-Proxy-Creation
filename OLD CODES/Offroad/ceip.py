"""CEIP / NIR national subsector shares (1A3c, 1A3ei, 1A3eii) and per-pixel broadcast."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ISO2_TO_ISO3: dict[str, str] = {"EL": "GRC", "GR": "GRC"}


def _norm_pol(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(s).strip().lower())


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    cols = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _parse_emission_value(v: Any) -> float:
    """Parse numeric emission; CEIP often uses NO / NE / IE for confidential or missing."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    s = str(v).strip().upper()
    if s in ("NA", "N/A", "C", "-", "", "NAN", "NO", "NE", "IE", "C"):
        return float("nan")
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _resolve_country_iso3(raw: Any, cntr_code_to_iso3: dict[str, str]) -> str | None:
    c = str(raw).strip().upper()
    if not c:
        return None
    if len(c) == 3 and c.isalpha():
        return c
    if len(c) == 2:
        return str(cntr_code_to_iso3.get(c, _ISO2_TO_ISO3.get(c, c))).upper()
    return str(cntr_code_to_iso3.get(c, c)).upper() if c in cntr_code_to_iso3 else None


def _offroad_sector_leg(sector_raw: Any) -> int | None:
    """
    Map CEIP/NFR-style sector cell to rail / pipeline / non-road leg index.

    Order: match longest SNAP token first (1A3eii before 1A3ei).
    """
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


def read_ceip_shares(
    xlsx_path: Path,
    *,
    sheet: str | None,
    pollutant_aliases: dict[str, str],
    pollutants_wanted: list[str],
    cntr_code_to_iso3: dict[str, str],
    default_triple: tuple[float, float, float],
    ceip_year: int | None = None,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """
    Return ``shares[pollutant_key][iso3_upper] = (s_rail, s_pipe, s_nonroad)``, each summing to ~1.

    Supports:

    * **Wide** workbook: columns country, pollutant, E_1A3c / E_1A3ei / E_1A3eii (or aliases).
    * **Long** CEIP-style sheet: COUNTRY, YEAR, SECTOR, POLLUTANT, TOTAL — sector differentiates
      1A3c (rail), 1A3ei (pipeline), 1A3eii (non-road); rows filtered by ``ceip_year`` when set.
    """
    if not xlsx_path.is_file():
        logger.warning("CEIP workbook missing (%s); using defaults only.", xlsx_path)
        iso = str(cntr_code_to_iso3.get("EL", "GRC")).upper()
        d_rail, d_pipe, d_nr = default_triple
        return {p: {iso: (d_rail, d_pipe, d_nr)} for p in pollutants_wanted}

    raw = pd.read_excel(xlsx_path, sheet_name=sheet if sheet else 0, header=0, engine="openpyxl")

    c_country = _pick_col(raw, ("country", "iso3", "country_code", "cntr_code", "iso", "c_country"))
    c_pol = _pick_col(raw, ("pollutant", "species", "compound"))
    c1 = _pick_col(raw, ("e_1a3c", "1a3c", "eea_1a3c"))
    c2 = _pick_col(raw, ("e_1a3ei", "1a3ei", "eea_1a3ei"))
    c3 = _pick_col(raw, ("e_1a3eii", "1a3eii", "eea_1a3eii"))

    cntr_map = dict(cntr_code_to_iso3)
    for k, v in list(_ISO2_TO_ISO3.items()):
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
        cc = _resolve_country_iso3(row[c_country], cntr_map)
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

    _fill_defaults(out, pollutants_wanted, cntr_map, default_triple)
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
    wanted = {_norm_pol(p) for p in pollutants_wanted}
    out: dict[str, dict[str, tuple[float, float, float]]] = {
        _norm_pol(p): {} for p in pollutants_wanted
    }

    df = raw.copy()
    year_filter = ceip_year
    if c_year is not None and year_filter is None:
        try:
            ser = pd.to_numeric(df[c_year], errors="coerce").dropna()
            if len(ser):
                year_filter = int(ser.mode().iloc[0])
                logger.info("CEIP Offroad: using modal sheet year %s (set defaults.ceip_year to override).", year_filter)
        except Exception:
            year_filter = None
    if c_year is not None and year_filter is not None:
        try:
            yn = pd.to_numeric(df[c_year], errors="coerce")
            df = df[yn == int(year_filter)]
        except Exception:
            pass
        if df.empty:
            logger.warning(
                "CEIP Offroad: no rows for year=%s; using full sheet.",
                year_filter,
            )
            df = raw.copy()

    # Accumulate emissions e[0]=1A3c e[1]=1A3ei e[2]=1A3eii per (iso3, pollutant_key)
    acc: dict[tuple[str, str], list[float]] = {}

    for _, row in df.iterrows():
        cc = _resolve_country_iso3(row[c_country], cntr_map)
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
        out.setdefault(polk, {})[iso3] = (
            trip[0] / et,
            trip[1] / et,
            trip[2] / et,
        )

    logger.info(
        "CEIP Offroad (long format): loaded shares for %d country×pollutant keys.",
        sum(len(v) for v in out.values()),
    )
    _fill_defaults(out, pollutants_wanted, cntr_map, default_triple)
    return out


def _fill_defaults(
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


def build_share_arrays(
    country_idx: np.ndarray,
    idx_to_iso: dict[int, str],
    shares: dict[str, dict[str, tuple[float, float, float]]],
    pollutant: str,
    fallback_iso: str,
    *,
    default_triple: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast ``(s_rail, s_pipe, s_nr)`` to grid using ``country_idx``."""
    pk = _norm_pol(pollutant)
    table = shares.get(pk, {})
    sr = np.zeros_like(country_idx, dtype=np.float32)
    sp = np.zeros_like(country_idx, dtype=np.float32)
    sn = np.zeros_like(country_idx, dtype=np.float32)
    fb = fallback_iso.strip().upper()

    def _fallback_trip() -> tuple[float, float, float]:
        if table.get(fb) is not None:
            return table[fb]
        if table:
            return next(iter(table.values()))
        return default_triple

    trip_fallback = _fallback_trip()

    for k, iso in idx_to_iso.items():
        if k <= 0:
            continue
        iso_u = str(iso).strip().upper()
        trip = table.get(iso_u) or table.get(fb)
        if trip is None:
            trip = trip_fallback if table else default_triple
        sr[country_idx == k] = np.float32(trip[0])
        sp[country_idx == k] = np.float32(trip[1])
        sn[country_idx == k] = np.float32(trip[2])

    mask = country_idx == 0
    trip0 = table.get(fb)
    if trip0 is None:
        trip0 = trip_fallback if table else default_triple
    sr[mask] = np.float32(trip0[0])
    sp[mask] = np.float32(trip0[1])
    sn[mask] = np.float32(trip0[2])
    return sr, sp, sn
