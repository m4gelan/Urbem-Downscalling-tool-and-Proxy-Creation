"""Parse K_Agriculture run-country filters."""
from __future__ import annotations

from typing import Any


def parse_run_country_codes(run: dict[str, Any] | None) -> frozenset[str] | None:
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
