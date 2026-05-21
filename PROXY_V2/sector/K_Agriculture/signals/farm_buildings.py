from __future__ import annotations

import numpy as np

from PROXY_V2.core import log
from PROXY_V2.sector.K_Agriculture.helper import AgReferenceGrid
from PROXY_V2.sector.K_Agriculture.signals.livestock_housing_pasture import LivestockHousingPastureResult


def build_farm_buildings(livestock: LivestockHousingPastureResult) -> tuple[np.ndarray, AgReferenceGrid]:
    """Farm buildings layer (same raster as livestock ``built``)."""
    log.info("--- K_Agriculture signal: farm buildings ---")
    h = livestock.built
    log.info(f"farm buildings: max={float(h.max()):.6g} sum={float(h.sum()):.6g}")
    return h, livestock.ref
