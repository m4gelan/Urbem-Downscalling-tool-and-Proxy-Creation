from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features as rio_features
from rasterio.enums import MergeAlg
from shapely.geometry import mapping

from proxy.core import log
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells
from proxy.dataset_loaders.load_osm import _clip_buffer_metric


def _read_otm_country(gpkg: Path, layer: str, iso3_col: str, iso3: str) -> gpd.GeoDataFrame:
    where = f'"{iso3_col}" = \'{iso3.strip().upper()}\''
    g = gpd.read_file(gpkg, layer=layer, where=where)
    if g.crs is None:
        raise ValueError(f"OTM layer {layer!r} has no CRS ({gpkg})")
    log.info(f"OTM: {len(g)} segments for {iso3}")
    return g


def _road_bucket(highway: Any, road_types: dict[str, Any]) -> str | None:
    h = str(highway).strip().lower()
    for name, rcfg in road_types.items():
        if h in {str(x).strip().lower() for x in rcfg["highways"]}:
            return str(name)
    return None


def compute_pi(gdf: gpd.GeoDataFrame, otm_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    hw_col = otm_cfg["highway_column"]
    vkm_sum_col = otm_cfg["vkm_sum_column"]
    vkm_cols = otm_cfg["vkm_columns"]
    road_types = otm_cfg["road_types"]
    g = gdf.copy()
    g["_bucket"] = g[hw_col].map(lambda h: _road_bucket(h, road_types))
    g = g[g["_bucket"].notna()].copy()
    pi: dict[str, dict[str, float]] = {}
    for rname in road_types:
        sub = g[g["_bucket"] == rname]
        by_class = {str(c): float(sub[str(col)].fillna(0).sum()) for c, col in vkm_cols.items()}
        total = float(sub[vkm_sum_col].fillna(0).sum())
        if total <= 0:
            total = sum(by_class.values())
        if total <= 0:
            raise ValueError(f"OTM Pi: no VKM for road type {rname!r}")
        pi[str(rname)] = {c: v / total for c, v in by_class.items()}
    return pi


def _length_weighted_aadt_raster(
    gdf: gpd.GeoDataFrame,
    *,
    aadt_col: str,
    len_col: str,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    all_touched: bool,
) -> np.ndarray:
    h, w = int(height), int(width)
    out = np.zeros((h, w), dtype=np.float32)
    if gdf.empty:
        return out
    g = gdf.to_crs(raster_crs)
    aadt = pd.to_numeric(g[aadt_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    length = pd.to_numeric(g[len_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    val_sum = np.zeros((h, w), dtype=np.float64)
    len_sum = np.zeros((h, w), dtype=np.float64)
    shapes_v = [
        (mapping(geom), float(a * ln))
        for geom, a, ln in zip(g.geometry, aadt, length)
        if geom is not None and not geom.is_empty and ln > 0
    ]
    shapes_l = [
        (mapping(geom), float(ln))
        for geom, ln in zip(g.geometry, length)
        if geom is not None and not geom.is_empty and ln > 0
    ]
    if not shapes_v:
        return out
    rio_features.rasterize(
        shapes_v, out=val_sum, transform=transform, fill=0.0, dtype=np.float64,
        all_touched=all_touched, merge_alg=MergeAlg.add,
    )
    rio_features.rasterize(
        shapes_l, out=len_sum, transform=transform, fill=0.0, dtype=np.float32,
        all_touched=all_touched, merge_alg=MergeAlg.add,
    )
    hit = len_sum > 0
    out[hit] = (val_sum[hit] / len_sum[hit]).astype(np.float32)
    return out


def _roads_mask_raster(
    gdf: gpd.GeoDataFrame,
    *,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    all_touched: bool,
) -> np.ndarray:
    h, w = int(height), int(width)
    out = np.zeros((h, w), dtype=np.float32)
    if gdf.empty:
        return out
    g = gdf.to_crs(raster_crs)
    shapes = [
        (mapping(geom), 1.0)
        for geom in g.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return out
    rio_features.rasterize(
        shapes, out=out, transform=transform, fill=0.0, dtype=np.float32,
        all_touched=all_touched,
    )
    return out


def load_otm_rasters(
    otm_gpkg: Path,
    country_profile: dict[str, str],
    otm_cfg: dict[str, Any],
    cams_cells: dict[int, dict[str, Any]],
    *,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pi, z-scored AADT, per-type road masks, and raw length-weighted AADT rasters."""
    iso3 = country_profile["ISO3"]
    raw = _read_otm_country(
        otm_gpkg, str(otm_cfg["layer"]), str(otm_cfg["iso3_column"]), iso3,
    )
    pi = compute_pi(raw, otm_cfg)
    inside = pixels_inside_cams_cells(height, width, transform, raster_crs, cams_cells)
    all_touched = bool(otm_cfg["rasterize"]["all_touched"])
    hw_col = otm_cfg["highway_column"]
    aadt_col = otm_cfg["aadt_column"]
    len_col = otm_cfg["length_column"]
    road_types = otm_cfg["road_types"]
    osm_like = {
        "metric_crs": otm_cfg["metric_crs"],
        "domain_clip_buffer_m": float(otm_cfg["domain_clip_buffer_m"]),
        "buffer_m": {"point": 0, "line": 0, "polygon": 0},
    }
    rasters: dict[str, np.ndarray] = {}
    roads_by_type: dict[str, np.ndarray] = {}
    aadt_raw: dict[str, np.ndarray] = {}
    g = raw.copy()
    g["_bucket"] = g[hw_col].map(lambda h: _road_bucket(h, road_types))
    for rname, rcfg in road_types.items():
        sub = g[g["_bucket"] == rname].copy()
        buf = float(rcfg["buffer_m"])
        sub = _clip_buffer_metric(
            sub, cams_cells, osm_like, ("LineString", "MultiLineString"), buf,
        )
        aadt = _length_weighted_aadt_raster(
            sub, aadt_col=aadt_col, len_col=len_col,
            height=height, width=width, transform=transform, raster_crs=raster_crs,
            all_touched=all_touched,
        )
        aadt_raw[str(rname)] = aadt
        rasters[str(rname)] = z_score_inside(aadt, inside, upper_quantile=0.99, rescale_to_01=True)
        roads_by_type[str(rname)] = _roads_mask_raster(
            sub, height=height, width=width, transform=transform,
            raster_crs=raster_crs, all_touched=all_touched,
        )
        log.info(
            f"OTM {rname}: {int((roads_by_type[rname] > 0).sum())} road px, "
            f"AADT z>0 {int((rasters[rname] > 0).sum())} px"
        )
    return pi, rasters, roads_by_type, aadt_raw
