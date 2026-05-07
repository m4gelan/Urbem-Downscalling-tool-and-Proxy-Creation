"""GNFR code tables for CAMS source-row categories.

CAMS enumerates F as F1..F4, so this module is the single source of truth for
mapping public GNFR letter codes to 1-based CAMS emission-category indices.
"""
from __future__ import annotations


GNFR_ORDER: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F1",
    "F2",
    "F3",
    "F4",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
)


def gnfr_code_to_index(code: str) -> int:
    """1-based index into :data:`GNFR_ORDER` for a CAMS GNFR code.

    ``F`` alone is rejected because CAMS enumerates F1..F4 explicitly.
    """
    c = str(code).strip().upper()
    if c == "F":
        raise ValueError("Use F1, F2, F3, or F4 (not F alone).")
    return GNFR_ORDER.index(c) + 1
