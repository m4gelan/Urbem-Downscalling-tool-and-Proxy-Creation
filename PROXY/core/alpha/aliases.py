"""Canonicalization of pollutant labels and country codes for alpha/CEIP lookups.

Kept minimal on purpose: this is the module every other alpha/CEIP loader is expected to
import when it needs to compare pollutant keys or resolve a country identifier, so there
is exactly one source of truth for naming.
"""
from __future__ import annotations

import re
from typing import Any


ISO2_TO_ISO3_CEIP: dict[str, str] = {"EL": "GRC", "GR": "GRC"}


def _norm_pol(s: Any) -> str:
    """Lowercase + strip non ``[a-z0-9_]`` (legacy behavior used for band-name keys)."""
    return re.sub(r"[^a-z0-9_]", "", str(s).strip().lower())


def norm_pollutant_key(label: Any) -> str:
    """Normalize pollutant labels for keys and band names (same as CEIP share tables)."""
    return _norm_pol(label)


def resolve_country_iso3(raw: Any, cntr_code_to_iso3: dict[str, str]) -> str | None:
    """ISO3 resolver used across CEIP loaders.

    - 3-letter alphabetic input is returned uppercased.
    - 2-letter input is mapped via ``cntr_code_to_iso3``, then ``ISO2_TO_ISO3_CEIP``.
    - Otherwise, looked up in ``cntr_code_to_iso3`` or returned ``None``.
    """
    c = str(raw).strip().upper()
    if not c:
        return None
    if len(c) == 3 and c.isalpha():
        return c
    if len(c) == 2:
        return str(cntr_code_to_iso3.get(c, ISO2_TO_ISO3_CEIP.get(c, c))).upper()
    return str(cntr_code_to_iso3.get(c, c)).upper() if c in cntr_code_to_iso3 else None


def normalize_country_alpha(country: str) -> str:
    """Country normalization used by ``compute_alpha`` (EL -> GR inventory code).

    This matches the `COUNTRY` column of the reported-emissions workbook, which uses GR
    (not GRC). Other pipelines that need ISO3 should use :func:`resolve_country_iso3`.
    """
    token = country.strip().upper()
    aliases = {"EL": "GR"}
    return aliases.get(token, token)


def normalize_country_token(country: str) -> str:
    """Alpha-CSV country normalization (EL -> GR), distinct from ISO3 for CAMS lookups."""
    c = str(country).strip().upper()
    return "GR" if c == "EL" else c
