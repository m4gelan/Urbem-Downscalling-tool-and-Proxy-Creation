from __future__ import annotations

import numpy as np

from proxy.core import log
from proxy.sector.K_Agriculture.helper import AgReferenceGrid
from proxy.sector.K_Agriculture.signals.livestock_housing_pasture import LivestockHousingPastureResult


def build_farm_buildings(livestock: LivestockHousingPastureResult) -> tuple[np.ndarray, AgReferenceGrid]:
    """Farm buildings layer (same raster as livestock ``built``)."""
    log.info("--- K_Agriculture signal: farm buildings ---")
    h = livestock.built
    log.info(f"farm buildings: max={float(h.max()):.6g} sum={float(h.sum()):.6g}")
    return h, livestock.ref
