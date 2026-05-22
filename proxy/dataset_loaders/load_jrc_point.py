from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from proxy.core import log

# JRC Open Power combustion ``type_g`` (same set as PROXY.core.matching).
JRC_COMBUSTION_TYPE_G: frozenset[str] = frozenset(
    {
        "Fossil Hard coal",
        "Fossil Brown coal/Lignite",
        "Fossil Peat",
        "Fossil Oil",
        "Fossil Oil shale",
        "Fossil Gas",
        "Fossil Coal-derived gas",
        "Biomass",
        "Waste",
    }
)


def _normalize_type_g(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    for c in JRC_COMBUSTION_TYPE_G:
        if s.lower() == c.lower():
            return c
    return s


def load_jrc_points(
    jrc_csv: Path,
    *,
    year: int,
    country_full_name: str,
) -> dict[str, dict[str, Any]]:
    """
    JRC Open Power units for point matching.

    Filters: ``country`` = *country_full_name*, combustion ``type_g`` only,
    drop rows with ``year_decommissioned`` < *year* (CAMS reference year).

    Returns ``{unit_id: {name_g, capacity_g, type_g, lat, lon}}``.
    """
    df = pd.read_csv(jrc_csv, low_memory=False)
    country = str(country_full_name).strip()
    df = df[df["country"].astype(str).str.strip().eq(country)].copy()

    tg = df["type_g"].map(_normalize_type_g)
    df = df[tg.isin(JRC_COMBUSTION_TYPE_G)].copy()

    ydec = pd.to_numeric(df["year_decommissioned"], errors="coerce")
    df = df[ydec.isna() | (ydec >= int(year))].copy()

    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    cap = pd.to_numeric(df["capacity_g"], errors="coerce")
    df = df.assign(_lat=lat, _lon=lon, _cap=cap).dropna(subset=["_lat", "_lon"])

    out: dict[str, dict[str, Any]] = {}
    for i, row in df.iterrows():
        uid = str(row["eic_g"]) if "eic_g" in df.columns else str(i)
        out[uid] = {
            "name_g": str(row["name_g"]),
            "capacity_g": float(row["_cap"]) if pd.notna(row["_cap"]) else 0.0,
            "type_g": str(row["type_g"]),
            "lat": float(row["_lat"]),
            "lon": float(row["_lon"]),
        }

    nout = len(out)
    log.info(f"Total JRC combustion units: {nout}")
    for j, (uid, row) in enumerate(out.items()):
        if j >= 1:
            break
        log.debug(
            f" Example unit: ID={uid} name={row['name_g']!r} "
            f"type={row['type_g']!r} capacity_g={row['capacity_g']:.1f} "
            f"lat={row['lat']:.4f} lon={row['lon']:.4f}"
        )
    log.debug(f"JRC load complete: {nout} units from {jrc_csv.name}")
    return out
