"""Sub-proxy: rail (OSM lines + landuse_railway) → coverage and z-score."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from PROXY.core.osm_corine_proxy import osm_coverage_fraction, z_score

from .common import lines_buffered_coverage
from .constants import BAD_RAILWAY, RAILWAY_LIFECYCLE_DISALLOW

logger = logging.getLogger(__name__)


def _parse_osm_tags(row: pd.Series) -> dict[str, str]:
    raw = row.get("osm_tags")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


def _railway_lifecycle_ok(tags: dict[str, str]) -> bool:
    for k in ("railway", "disused:railway"):
        v = tags.get(k)
        if v and str(v).strip().lower() in RAILWAY_LIFECYCLE_DISALLOW:
            return False
    return True


def filter_rail_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    keep: list[bool] = []
    for _, row in gdf.iterrows():
        tags = _parse_osm_tags(row)
        rw = tags.get("railway") or row.get("railway")
        if rw is None:
            rw = ""
        rs = str(rw).strip().lower()
        if rs in BAD_RAILWAY:
            keep.append(False)
            continue
        if not _railway_lifecycle_ok(tags):
            keep.append(False)
            continue
        keep.append(True)
    return gdf.loc[keep].copy()


def load_landuse_railway_polygons(osm_gpkg: Path) -> gpd.GeoDataFrame:
    try:
        ar = gpd.read_file(osm_gpkg, layer="osm_offroad_areas")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    if ar.empty or "offroad_family" not in ar.columns:
        return gpd.GeoDataFrame(geometry=[], crs=ar.crs)
    m = ar["offroad_family"].astype(str) == "landuse_railway"
    out = ar.loc[m].copy()
    return out if not out.empty else gpd.GeoDataFrame(geometry=[], crs=ar.crs)


def build_rail_coverage_and_z(
    osm_gpkg: Path,
    ref: dict[str, Any],
    *,
    rail_buffer_m: float,
    osm_subdivide: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw fractional rail footprint and z-scored proxy."""
    try:
        rlines = gpd.read_file(osm_gpkg, layer="osm_offroad_rail_lines")
    except Exception:
        rlines = gpd.GeoDataFrame(geometry=[], crs=None)
    rlines = filter_rail_lines(rlines)
    lrail = load_landuse_railway_polygons(osm_gpkg)
    rail_cov = lines_buffered_coverage(rlines, ref, rail_buffer_m, osm_subdivide)
    if not lrail.empty:
        rail_cov = np.clip(
            rail_cov
            + osm_coverage_fraction(
                lrail.to_crs(ref["crs"]),
                ref,
                subdivide_factor=osm_subdivide,
            ),
            0.0,
            1.0,
        )
    z_rail = z_score(rail_cov.astype(np.float64))
    return rail_cov.astype(np.float32), z_rail
