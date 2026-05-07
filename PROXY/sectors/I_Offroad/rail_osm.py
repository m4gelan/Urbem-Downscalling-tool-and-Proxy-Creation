"""Rail (1A3c): OSM lines (diesel vs electric weights) + railway yard polygons."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from PROXY.core.osm_corine_proxy import osm_coverage_fraction, z_score

from PROXY.core.osm_lines import lines_buffered_coverage

logger = logging.getLogger(__name__)


def _parse_osm_tags(row: pd.Series) -> dict[str, str]:
    raw = row.get("osm_tags")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {}


def _railway_lifecycle_ok(
    tags: dict[str, str],
    *,
    lifecycle_disallow: frozenset[str],
) -> bool:
    for k in ("railway", "disused:railway"):
        v = tags.get(k)
        if v and str(v).strip().lower() in lifecycle_disallow:
            return False
    return True


def filter_rail_lines(
    gdf: gpd.GeoDataFrame,
    *,
    bad_line_types: frozenset[str],
    lifecycle_disallow: frozenset[str],
    exclude_service_siding: bool = False,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    keep: list[bool] = []
    for _, row in gdf.iterrows():
        tags = _parse_osm_tags(row)
        if exclude_service_siding:
            sv = tags.get("service") or row.get("service")
            if sv is not None and str(sv).strip().lower() == "siding":
                keep.append(False)
                continue
        rw = tags.get("railway") or row.get("railway")
        if rw is None:
            rw = ""
        rs = str(rw).strip().lower()
        if rs in bad_line_types:
            keep.append(False)
            continue
        if not _railway_lifecycle_ok(tags, lifecycle_disallow=lifecycle_disallow):
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


def _split_diesel_electric_rail(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split rail line geometries into diesel vs electric using ``rail_propulsion`` (or tags)."""
    if gdf.empty:
        return gdf.iloc[0:0].copy(), gdf.iloc[0:0].copy()
    diesel_idx: list[Any] = []
    electric_idx: list[Any] = []
    for idx, row in gdf.iterrows():
        rp = row.get("rail_propulsion")
        if rp is None or (isinstance(rp, float) and pd.isna(rp)) or str(rp).strip() == "":
            rp = _parse_osm_tags(row).get("rail_propulsion")
        is_electric = str(rp).strip().lower() == "electric"
        (electric_idx if is_electric else diesel_idx).append(idx)
    diesel = gdf.loc[diesel_idx].copy() if diesel_idx else gdf.iloc[0:0].copy()
    electric = gdf.loc[electric_idx].copy() if electric_idx else gdf.iloc[0:0].copy()
    return diesel, electric


def build_rail_coverage_and_z(
    osm_gpkg: Path,
    ref: dict[str, Any],
    *,
    rail_buffer_m: float,
    osm_subdivide: int,
    bad_line_types: frozenset[str],
    lifecycle_disallow: frozenset[str],
    proxy_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Raw fractional rail footprint (weighted diesel/electric line buffers + yards) and z-scored proxy.

    ``proxy_cfg`` optional keys: ``w_rail_diesel`` (1.0), ``w_rail_electric`` (0.15),
    ``w_rail_yard`` (1.0), ``rail_exclude_service_siding`` (false).
    """
    pc = dict(proxy_cfg or {})
    w_diesel = float(pc.get("w_rail_diesel", 1.0))
    w_electric = float(pc.get("w_rail_electric", 0.15))
    w_yard = float(pc.get("w_rail_yard", 1.0))
    exclude_siding = bool(pc.get("rail_exclude_service_siding", False))

    try:
        rlines = gpd.read_file(osm_gpkg, layer="osm_offroad_rail_lines")
    except Exception:
        rlines = gpd.GeoDataFrame(geometry=[], crs=None)
    rlines = filter_rail_lines(
        rlines,
        bad_line_types=bad_line_types,
        lifecycle_disallow=lifecycle_disallow,
        exclude_service_siding=exclude_siding,
    )
    h, w = int(ref["height"]), int(ref["width"])
    rail_cov = np.zeros((h, w), dtype=np.float64)
    diesel_gdf, electric_gdf = _split_diesel_electric_rail(rlines)
    buf = float(rail_buffer_m)
    sub = int(osm_subdivide)
    if not diesel_gdf.empty and w_diesel != 0.0:
        rail_cov += w_diesel * lines_buffered_coverage(diesel_gdf, ref, buf, sub).astype(np.float64)
    if not electric_gdf.empty and w_electric != 0.0:
        rail_cov += w_electric * lines_buffered_coverage(electric_gdf, ref, buf, sub).astype(np.float64)

    lrail = load_landuse_railway_polygons(osm_gpkg)
    if not lrail.empty and w_yard != 0.0:
        crs = rasterio.crs.CRS.from_string(ref["crs"])
        rail_cov += w_yard * osm_coverage_fraction(
            lrail.to_crs(crs),
            ref,
            subdivide_factor=sub,
        ).astype(np.float64)

    rail_cov = np.clip(rail_cov, 0.0, 1.0)
    z_rail = z_score(rail_cov.astype(np.float64))
    return rail_cov.astype(np.float32), z_rail
