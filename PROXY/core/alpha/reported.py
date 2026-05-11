"""Shared helpers for reported-emissions/CEIP ingestion."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# ISO-3166-1 alpha-2 -> alpha-3 fallback table for CEIP COUNTRY parsing.
ISO2_TO_ISO3_EU: dict[str, str] = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "HR": "HRV",
    "CY": "CYP",
    "CZ": "CZE",
    "DK": "DNK",
    "EE": "EST",
    "FI": "FIN",
    "FR": "FRA",
    "DE": "DEU",
    "EL": "GRC",
    "GR": "GRC",
    "HU": "HUN",
    "IE": "IRL",
    "IT": "ITA",
    "LV": "LVA",
    "LT": "LTU",
    "LU": "LUX",
    "MT": "MLT",
    "NL": "NLD",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SK": "SVK",
    "SI": "SVN",
    "ES": "ESP",
    "SE": "SWE",
    "UK": "GBR",
    "GB": "GBR",
    "NO": "NOR",
    "CH": "CHE",
    "IS": "ISL",
    "LI": "LIE",
    "AL": "ALB",
}


def short_country(iso3: str, aliases: dict[str, str] | None = None) -> str:
    c = str(iso3 or "").strip().upper()
    if aliases and c in aliases:
        return str(aliases[c]).strip().upper()
    return c


def normalize_sector_token(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).strip().upper())


def normalize_inventory_sector(s: Any) -> str:
    """Canonical normalization for inventory sector / NFR tokens (workbook + CEIP tables)."""
    return normalize_sector_token(str(s))


def parse_float_or_nan(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    s = str(v).strip().upper()
    if s in ("NA", "N/A", "C", "-", "", "NAN"):
        return float("nan")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return float("nan")


def to_iso3(code: str, aliases: dict[str, str] | None = None) -> str:
    c = str(code).strip().upper()
    if len(c) == 3 and c.isalpha():
        return c
    merged = dict(ISO2_TO_ISO3_EU)
    if aliases:
        merged.update({str(k).upper(): str(v).upper() for k, v in aliases.items()})
    if len(c) == 2:
        return merged.get(c, c)
    return c


def resolve_iso3_reported(country_raw: str, cntr_code_to_iso3: dict[str, str]) -> str | None:
    c = str(country_raw).strip().upper()
    if not c:
        return None
    if len(c) == 3 and c.isalpha():
        return c
    merged = dict(ISO2_TO_ISO3_EU)
    merged.update({str(k).upper(): str(v).upper() for k, v in (cntr_code_to_iso3 or {}).items()})
    if len(c) == 2:
        return merged.get(c)
    if c in merged:
        return merged[c]
    return None


EU_AGGREGATE_CODES = frozenset({"EU11", "EU27"})


def resolve_reporting_key(country_raw: Any, cntr_code_to_iso3: dict[str, str]) -> str | None:
    """Map workbook ``COUNTRY`` to either ISO3 or an EU aggregate code (``EU11``, ``EU27``)."""
    c = str(country_raw).strip().upper()
    if not c:
        return None
    if c in EU_AGGREGATE_CODES:
        return c
    return resolve_iso3_reported(country_raw, cntr_code_to_iso3)
