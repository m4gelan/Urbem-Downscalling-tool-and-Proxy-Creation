"""Rasterize OSM features matching CEIP-style ``any_of`` tag rules onto the reference grid."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio import features


def _row_matches_any_of(row: Any, any_of: list[dict[str, str]]) -> bool:
    for rule in any_of:
        if not isinstance(rule, dict):
            continue
        for col, val in rule.items():
            want_c = str(col).strip().lower()
            want_v = str(val).strip().lower()
            for c in row.index:
                if str(c).strip().lower() != want_c:
                    continue
                cell = row[c]
                if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                    continue
                if str(cell).strip().lower() == want_v:
                    return True
                break
    return False


def filter_osm_gdf_by_rules(gdf: gpd.GeoDataFrame, osm_rules: dict[str, Any]) -> gpd.GeoDataFrame:
    """Keep rows matching ``osm_rules['any_of']`` list of single-key dicts."""
    if gdf.empty:
        return gdf
    any_of = osm_rules.get("any_of")
    if not isinstance(any_of, list) or not any_of:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    keep: list[int] = []
    for i in range(len(gdf)):
        if _row_matches_any_of(gdf.iloc[i], any_of):
            keep.append(i)
    if not keep:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gdf.iloc[keep].copy()


def rasterize_osm_binary_mask(
    gdf: gpd.GeoDataFrame,
    *,
    transform: Any,
    height: int,
    width: int,
    crs: Any,
) -> np.ndarray:
    """
    Return ``(H, W)`` float array with 1.0 where OSM geometry covers pixel, else 0.
    """
    out = np.zeros((int(height), int(width)), dtype=np.float64)
    if gdf.empty:
        return out
    gg = gdf.to_crs(crs)
    shapes = []
    for geom in gg.geometry:
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, 1.0))
    if not shapes:
        return out
    features.rasterize(
        shapes,
        out=out,
        transform=transform,
        fill=0.0,
        default_value=1.0,
        all_touched=True,
    )
    return out
