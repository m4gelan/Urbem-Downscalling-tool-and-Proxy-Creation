"""Shared rasterization helpers (buffered lines, etc.)."""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio

from PROXY.core.osm_corine_proxy import osm_coverage_fraction

logger = logging.getLogger(__name__)


def lines_buffered_coverage(
    gdf: gpd.GeoDataFrame,
    ref: dict[str, Any],
    buffer_m: float,
    subdivide: int,
) -> np.ndarray:
    """Buffered line geometries → fractional coverage per fine pixel."""
    if gdf.empty:
        h, w = int(ref["height"]), int(ref["width"])
        return np.zeros((h, w), dtype=np.float32)
    crs = rasterio.crs.CRS.from_string(ref["crs"])
    g = gdf.to_crs(crs)
    geom_ok = g.geometry.notna() & (~g.geometry.is_empty)
    g = g.loc[geom_ok].copy()
    if g.empty:
        h, w = int(ref["height"]), int(ref["width"])
        return np.zeros((h, w), dtype=np.float32)
    buf = float(buffer_m)
    try:
        gb = g.buffer(buf)
        gp = gpd.GeoDataFrame(geometry=gb, crs=g.crs)
    except Exception as exc:
        logger.warning("buffer failed (%s); using raw geometries", exc)
        gp = g
    return osm_coverage_fraction(gp, ref, subdivide_factor=max(1, int(subdivide)))
