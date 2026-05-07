"""Sub-proxy: pipelines (OSM lines + pipeline polygons) → coverage and z-score."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from PROXY.core.osm_corine_proxy import z_score
from Waste.j_waste_weights.io_utils import warp_raster_to_ref

from .common import lines_buffered_coverage

logger = logging.getLogger(__name__)


def load_pipeline_union(osm_gpkg: Path, _root: Path) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for layer in ("osm_offroad_pipeline_lines",):
        try:
            x = gpd.read_file(osm_gpkg, layer=layer)
            if not x.empty:
                frames.append(x)
        except Exception as exc:
            logger.warning("pipeline layer %s: %s", layer, exc)
    try:
        ar = gpd.read_file(osm_gpkg, layer="osm_offroad_areas")
        if not ar.empty:
            fam = ar.get("offroad_family")
            if fam is not None:
                ar = ar.loc[fam.astype(str) == "man_made_pipeline"].copy()
            else:
                ar = ar.iloc[0:0]
            if not ar.empty:
                frames.append(ar)
    except Exception as exc:
        logger.warning("areas layer for pipeline: %s", exc)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    out = pd.concat(frames, ignore_index=True)
    crs = frames[0].crs
    return gpd.GeoDataFrame(out, geometry=out.geometry, crs=crs)


def build_pipeline_raw_z(
    osm_gpkg: Path,
    root: Path,
    ref: dict[str, Any],
    *,
    pipeline_buffer_m: float,
    osm_subdivide: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage along buffered pipelines and z-score (before facility blend)."""
    plines = load_pipeline_union(osm_gpkg, root)
    pipe_cov = lines_buffered_coverage(plines, ref, pipeline_buffer_m, osm_subdivide)
    z_pipe_raw = z_score(pipe_cov.astype(np.float64))
    return pipe_cov.astype(np.float32), z_pipe_raw


def blend_pipeline_with_facilities(
    *,
    root: Path,
    ref: dict[str, Any],
    z_pipe_raw: np.ndarray,
    paths: dict[str, Any],
    pipe_fac_weight: float,
    pipe_raw_weight: float,
) -> np.ndarray:
    """Optional facilities raster blend (YAML paths.facilities_tif)."""
    fac_path = paths.get("facilities_tif")
    if not fac_path:
        return z_pipe_raw.astype(np.float32)
    fac_p = Path(fac_path)
    if not fac_p.is_absolute():
        fac_p = root / fac_p
    if not fac_p.is_file():
        return z_pipe_raw.astype(np.float32)
    fac_arr = warp_raster_to_ref(fac_p, ref)
    z_fac = z_score(np.nan_to_num(fac_arr, nan=0.0))
    out = pipe_fac_weight * z_pipe_raw + pipe_raw_weight * z_fac
    return out.astype(np.float32)


def build_pipeline_z_final(
    osm_gpkg: Path,
    root: Path,
    ref: dict[str, Any],
    paths: dict[str, Any],
    proxy_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns ``(z_pipe_final, z_pipe_raw, pipe_cov)`` — final z may include facility blend.
    ``proxy_cfg`` uses keys: pipeline_buffer_m, osm_subdivide, pipe_fac_weight, pipe_raw_weight.
    """
    buf = float(proxy_cfg.get("pipeline_buffer_m", 75))
    subdiv = int(proxy_cfg.get("osm_subdivide", 4))
    fac_w = float(proxy_cfg.get("pipe_fac_weight", 0.8))
    raw_w = float(proxy_cfg.get("pipe_raw_weight", 0.2))
    pipe_cov, z_raw = build_pipeline_raw_z(
        osm_gpkg, root, ref, pipeline_buffer_m=buf, osm_subdivide=subdiv
    )
    z_final = blend_pipeline_with_facilities(
        root=root,
        ref=ref,
        z_pipe_raw=z_raw,
        paths=paths,
        pipe_fac_weight=fac_w,
        pipe_raw_weight=raw_w,
    )
    return z_final, z_raw.astype(np.float32), pipe_cov
