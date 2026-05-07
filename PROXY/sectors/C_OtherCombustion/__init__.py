"""GNFR C other combustion — see ``README.md``; run via ``pipeline``."""

from __future__ import annotations

from .pipeline import run, run_downscale, run_other_combustion_weight_build

SECTOR_KEY = "C_OtherCombustion"

__all__ = ["SECTOR_KEY", "run", "run_downscale", "run_other_combustion_weight_build"]
