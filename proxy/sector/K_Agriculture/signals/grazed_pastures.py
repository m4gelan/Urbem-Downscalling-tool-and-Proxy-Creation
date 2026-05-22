from __future__ import annotations

import numpy as np

from proxy.core import log
from proxy.sector.K_Agriculture.helper import AgReferenceGrid
from proxy.sector.K_Agriculture.signals.livestock_housing_pasture import LivestockHousingPastureResult


def build_grazed_pastures(livestock: LivestockHousingPastureResult) -> tuple[np.ndarray, AgReferenceGrid]:
    """Signal 3 — grazed pasture layer (same raster as livestock ``grazing``)."""
    log.info("--- K_Agriculture signal: grazed pastures ---")
    g = livestock.grazing
    log.info(f"grazed pastures: max={float(g.max()):.6g} sum={float(g.sum()):.6g}")
    return g, livestock.ref
