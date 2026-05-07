"""Parse run.country for single- or multi-country pipeline runs."""

from __future__ import annotations

from typing import Any


def parse_run_country_codes(run: dict[str, Any] | None) -> frozenset[str] | None:
    """
    Return country codes to keep (NUTS CNTR_CODE / LUCAS POINT_NUTS0), or None = no filter (all).

    Accepts:
    - missing / empty string: all countries
    - a string: one code, or comma-separated codes (e.g. "DE, FR, EL")
    - a list of strings: e.g. ["DE", "FR", "EL"]
    """
    if not run:
        return None
    raw = run.get("country")
    if raw is None:
        return None
    if isinstance(raw, list):
        codes = [str(x).strip().upper() for x in raw if str(x).strip()]
        return frozenset(codes) if codes else None
    s = str(raw).strip()
    if not s:
        return None
    if "," in s:
        parts = [p.strip().upper() for p in s.split(",") if p.strip()]
        return frozenset(parts) if parts else None
    return frozenset([s.upper()])
