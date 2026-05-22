from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from proxy.core import log


def _normalize_eprtr_sector_codes(eprtr_sector_code: int | Sequence[int]) -> set[int]:
    """Accept a single sector code or a non-empty sequence (YAML may give int or list)."""
    if isinstance(eprtr_sector_code, str):
        raise TypeError("eprtr_sector_code must be int or sequence of ints, not str")
    if isinstance(eprtr_sector_code, Sequence):
        codes = {int(x) for x in eprtr_sector_code}
    else:
        codes = {int(eprtr_sector_code)}
    if not codes:
        raise ValueError("eprtr_sector_code must be a non-empty int or sequence of ints")
    return codes


def load_eprtr_points(
    eprtr_csv: Path,
    *,
    reporting_years: list[int],
    country_full_name: str,
    eprtr_sector_code: int | Sequence[int],
    eprtr_sub_sector_codes: list[str],
) -> dict[str, dict[str, Any]]:
    """
    EPRTR air-release facilities for point-matching fallback.

    Filters: ``countryName``, ``reportingYear`` in *reporting_years*,
    ``EPRTR_SectorCode`` in *eprtr_sector_code* (single int or list of ints),
    ``EPRTRAnnexIMainActivity`` in *eprtr_sub_sector_codes*, ``TargetRelease`` = AIR.
    One row per ``FacilityInspireId`` (latest reporting year).
    """
    years = {int(y) for y in reporting_years}
    sectors = _normalize_eprtr_sector_codes(eprtr_sector_code)
    sub_sector_codes = [str(code).strip() for code in eprtr_sub_sector_codes]
    country = str(country_full_name).strip()

    df = pd.read_csv(eprtr_csv, low_memory=False)

    df["reportingYear"] = pd.to_numeric(df["reportingYear"], errors="coerce")
    df["EPRTR_SectorCode"] = pd.to_numeric(df["EPRTR_SectorCode"], errors="coerce")

    df = df[
        df["countryName"].astype(str).str.strip().eq(country)
        & df["reportingYear"].isin(years)
        & df["EPRTR_SectorCode"].isin(sectors)
        & df["EPRTRAnnexIMainActivity"].astype(str).str.strip().isin(sub_sector_codes)
    ]

    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    
    df = df.assign(_lon=lon, _lat=lat).dropna(subset=["_lon", "_lat", "FacilityInspireId"])

    df = df.sort_values("reportingYear").groupby("FacilityInspireId", as_index=False).tail(1)

    log.debug(
        f"EPRTR load: country={country!r} sectors={sorted(sectors)} annex={sub_sector_codes!r} years={sorted(years)} | "
        f"{len(df)} facilities retained"
    )

    out: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        fid = str(row["FacilityInspireId"]).strip()
        if not fid or fid in out:
            continue

        out[fid] = {
            "facility_name": str(row["facilityName"]),
            "facility_id": fid,
            "lat": float(row["_lat"]),
            "lon": float(row["_lon"]),
            "reporting_year": int(row["reportingYear"]),
            "eprtr_annex": str(row["EPRTRAnnexIMainActivity"]),
        }

    return out


def load_eprtr_points_energy(
    lcp_csv: Path, *, 
    country_full_name: str) -> dict[str, dict[str, Any]]:
    """
    E-PRTR energy facilities LCP.

    Filters: ``countryName``
    """
    
    country = str(country_full_name).strip()
    log.info(lcp_csv)

    df = pd.read_csv(lcp_csv, sep=";", low_memory=False)

    df = df[df["countryName"].astype(str).str.strip().eq(country)]

    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    lat = pd.to_numeric(df["Latitude"], errors="coerce")

    df = df.assign(_lon=lon, _lat=lat).dropna(subset=["_lon", "_lat"])

    df = df.sort_values("reportingYear").groupby("installationPartName", as_index=False).tail(1)

    log.debug(
        f"EPRTR load: country={country!r} | {len(df)} facilities retained\n"
    )

    out: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        iname = str(row["installationPartName"]).strip()
        if not iname or iname in out:
            continue
        out[iname] = {
            "installation_part_name": iname,
            "lat": float(row["_lat"]),
            "lon": float(row["_lon"]),
        }

    return out
    