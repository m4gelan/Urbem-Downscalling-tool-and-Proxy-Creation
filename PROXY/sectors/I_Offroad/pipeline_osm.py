"""Pipeline (1A3ei): hydrocarbon lines + compressor facilities, blended proxy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.enums import MergeAlg
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping

from PROXY.core.osm_corine_proxy import z_score
from PROXY.core.osm_lines import lines_buffered_coverage
from PROXY.core.raster import warp_raster_to_ref

logger = logging.getLogger(__name__)


def load_pipeline_lines(osm_gpkg: Path) -> gpd.GeoDataFrame:
    try:
        x = gpd.read_file(osm_gpkg, layer="osm_offroad_pipeline_lines")
        return x if not x.empty else gpd.GeoDataFrame(geometry=[], crs=x.crs)
    except Exception as exc:
        logger.warning("pipeline lines layer: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs=None)


def load_pipeline_facilities(osm_gpkg: Path) -> gpd.GeoDataFrame:
    try:
        x = gpd.read_file(osm_gpkg, layer="osm_offroad_pipeline_facilities")
        return x if not x.empty else gpd.GeoDataFrame(geometry=[], crs=x.crs)
    except Exception as exc:
        logger.warning("pipeline facilities layer: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs=None)


def load_pipeline_union(osm_gpkg: Path, _root: Path) -> gpd.GeoDataFrame:
    """Hydrocarbon pipeline ways plus compressor features (for diagnostics / viz)."""
    frames: list[gpd.GeoDataFrame] = []
    for layer in ("osm_offroad_pipeline_lines", "osm_offroad_pipeline_facilities"):
        try:
            x = gpd.read_file(osm_gpkg, layer=layer)
            if not x.empty:
                frames.append(x)
        except Exception as exc:
            logger.warning("pipeline layer %s: %s", layer, exc)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    out = pd.concat(frames, ignore_index=True)
    crs = frames[0].crs
    return gpd.GeoDataFrame(out, geometry=out.geometry, crs=crs)


def _facilities_as_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Represent polygons/lines as points for facility kernels (spec: treat as point)."""
    if gdf.empty:
        return gdf
    geoms: list[Any] = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        gt = g.geom_type
        if gt == "Point":
            geoms.append(g)
        elif gt == "MultiPoint":
            geoms.extend(list(g.geoms))
        else:
            geoms.append(g.representative_point())
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(geometry=geoms, crs=gdf.crs)


def _facility_coverage_gaussian(
    gdf_pts: gpd.GeoDataFrame,
    ref: dict[str, Any],
    sigma_m: float,
) -> np.ndarray:
    """Unit impulses at compressor locations, Gaussian blur with sigma in metres."""
    h, w = int(ref["height"]), int(ref["width"])
    tr = ref["transform"]
    if gdf_pts.empty:
        return np.zeros((h, w), dtype=np.float64)
    crs = ref["crs"]
    g = gdf_pts.to_crs(crs)
    shapes: list[tuple[Any, float]] = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        shapes.append((mapping(geom), 1.0))
    if not shapes:
        return np.zeros((h, w), dtype=np.float64)
    burned = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=tr,
        fill=0,
        dtype=np.float32,
        merge_alg=MergeAlg.add,
    ).astype(np.float64)
    px_w = abs(float(tr[0]))
    px_h = abs(float(tr[4]))
    sig = float(sigma_m)
    sx = sig / px_w if px_w > 0 else 1.0
    sy = sig / px_h if px_h > 0 else 1.0
    blurred = gaussian_filter(burned, sigma=(sy, sx), mode="constant", cval=0.0)
    m = float(np.max(blurred)) if blurred.size else 0.0
    if m > 0:
        blurred /= m
    return blurred


def _facility_coverage_disc(
    gdf_pts: gpd.GeoDataFrame,
    ref: dict[str, Any],
    radius_m: float,
    osm_subdivide: int,
) -> np.ndarray:
    """Circular footprint around each facility (buffer rasterisation)."""
    if gdf_pts.empty:
        h, w = int(ref["height"]), int(ref["width"])
        return np.zeros((h, w), dtype=np.float64)
    return lines_buffered_coverage(
        gdf_pts,
        ref,
        float(radius_m),
        int(osm_subdivide),
    ).astype(np.float64)


def build_facility_coverage(
    osm_gpkg: Path,
    ref: dict[str, Any],
    proxy_cfg: dict[str, Any],
) -> np.ndarray:
    """Compressor / compression-substation coverage for blending with pipeline lines."""
    fac = load_pipeline_facilities(osm_gpkg)
    fac_pts = _facilities_as_points(fac)
    if fac_pts.empty:
        h, w = int(ref["height"]), int(ref["width"])
        return np.zeros((h, w), dtype=np.float64)

    mode = str(proxy_cfg.get("pipeline_facility_raster_mode", "gaussian")).strip().lower()
    sigma_m = float(proxy_cfg.get("pipeline_facility_sigma_m", 500.0))
    disc_m = float(proxy_cfg.get("pipeline_facility_disc_radius_m", 1000.0))
    subdiv = int(proxy_cfg.get("osm_subdivide", 4))

    if mode == "disc":
        return _facility_coverage_disc(fac_pts, ref, disc_m, subdiv)
    return _facility_coverage_gaussian(fac_pts, ref, sigma_m)


def build_pipeline_raw_z(
    osm_gpkg: Path,
    root: Path,
    ref: dict[str, Any],
    *,
    pipeline_buffer_m: float,
    osm_subdivide: int,
    proxy_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    pipe_cov_display, z_pipe_raw, cov_lines, cov_fac, z_facilities
        ``pipe_cov_display`` is a blended raw coverage for logging; ``z_pipe_raw`` is the OSM
        pipeline score (optionally mixed lines / facilities, then optional facilities GeoTIFF blend).
    """
    pc = dict(proxy_cfg)
    buf = float(pipeline_buffer_m)
    subdiv = int(osm_subdivide)
    plines = load_pipeline_lines(osm_gpkg)
    cov_lines = lines_buffered_coverage(plines, ref, buf, subdiv).astype(np.float64)
    z_lines = z_score(cov_lines.astype(np.float64))

    cov_fac = build_facility_coverage(osm_gpkg, ref, pc)
    has_fac = float(np.max(cov_fac)) > 1e-18
    z_fac = z_score(cov_fac.astype(np.float64)) if has_fac else np.zeros_like(z_lines, dtype=np.float32)

    w_line = float(pc.get("w_pipeline_lines_in_proxy", 0.30))
    w_fac = float(pc.get("w_pipeline_facilities_in_proxy", 0.70))
    if has_fac:
        z_pipe_raw = (w_line * z_lines + w_fac * z_fac).astype(np.float32)
        pipe_cov_display = (w_line * cov_lines + w_fac * cov_fac).astype(np.float32)
    else:
        z_pipe_raw = z_lines.astype(np.float32)
        pipe_cov_display = cov_lines.astype(np.float32)

    return (
        pipe_cov_display.astype(np.float32),
        z_pipe_raw.astype(np.float32),
        cov_lines.astype(np.float32),
        cov_fac.astype(np.float32),
        z_fac.astype(np.float32),
    )


def blend_pipeline_with_facilities(
    *,
    root: Path,
    ref: dict[str, Any],
    z_pipe_raw: np.ndarray,
    paths: dict[str, Any],
    pipe_fac_weight: float,
    pipe_raw_weight: float,
) -> np.ndarray:
    """Optional extra blend with ``facilities_tif`` (EPRTR / external raster)."""
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
    Returns ``(z_pipe_final, z_pipe_raw, pipe_cov)`` — OSM lines/facility blend, then optional
    ``facilities_tif`` blend. ``proxy_cfg`` uses pipeline_buffer_m, osm_subdivide, pipe_fac_weight,
    pipe_raw_weight, and facility kernel settings.
    """
    buf = float(proxy_cfg.get("pipeline_buffer_m", 75))
    subdiv = int(proxy_cfg.get("osm_subdivide", 4))
    fac_w = float(proxy_cfg.get("pipe_fac_weight", 0.8))
    raw_w = float(proxy_cfg.get("pipe_raw_weight", 0.2))
    pipe_cov, z_raw, _cov_l, _cov_f, _z_f = build_pipeline_raw_z(
        osm_gpkg, root, ref, pipeline_buffer_m=buf, osm_subdivide=subdiv, proxy_cfg=proxy_cfg
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
