"""UECT output helpers shared by UrbEm pipelines."""

from __future__ import annotations

from dataclasses import dataclass


UECT_POLLUTANTS_ORDER = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]


@dataclass(frozen=True)
class UectPointDefaults:
    Hi: int = -999
    Vi: int = -999
    Ti: int = -999
    radi: int = -999
