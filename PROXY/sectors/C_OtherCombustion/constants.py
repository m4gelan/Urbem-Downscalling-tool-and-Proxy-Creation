"""Shared constants for GNFR C other-combustion (model axis K=7)."""

from __future__ import annotations

MODEL_CLASSES: tuple[str, ...] = (
    "R_FIREPLACE",
    "R_HEATING_STOVE",
    "R_COOKING_STOVE",
    "R_BOILER_MAN",
    "R_BOILER_AUT",
    "C_BOILER_MAN",
    "C_BOILER_AUT",
)

CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(MODEL_CLASSES)}

END_USE_COMMERCIAL = "commercial"
