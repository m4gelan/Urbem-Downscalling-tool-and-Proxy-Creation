from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells


def _row_uid(row: Any, fallback: str) -> str:
    for col in ("uwwInspireIDFacility", "uwwCode", "OBJECTID"):
        if col not in row.index:
            continue
        v = row[col]
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        s = str(v).strip()
        if s:
            return s.replace(" ", "_")
    return fallback


def _read_uwwtd_plants_gdf(
    uwwtd_treatment_plants_filepath: Path,
    *,
    rpt_state_key: str,
    active_only: bool,
) -> gpd.GeoDataFrame:
    if not uwwtd_treatment_plants_filepath.is_file():
        log.warning(f"UWWTD treatment plants file missing: {uwwtd_treatment_plants_filepath}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.read_file(uwwtd_treatment_plants_filepath, datetime_as_string=True)
    if gdf.empty:
        return gdf

    key = str(rpt_state_key).strip().upper()
    if "rptMStateKey" not in gdf.columns:
        log.warning("UWWTD plants GPKG has no rptMStateKey column")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    rk = gdf["rptMStateKey"].astype(str).str.strip().str.upper()
    gdf = gdf[rk.eq(key)].copy()
    if gdf.empty:
        log.info(f"UWWTD treatment plants: no rows for rptMStateKey={key!r}")
        return gdf

    if active_only and "uwwState" in gdf.columns:
        st = pd.to_numeric(gdf["uwwState"], errors="coerce")
        gdf = gdf[st.fillna(0).eq(1)]

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _read_uwwtd_agg_gdf(
    uwwtd_agglomerations_filepath: Path,
    *,
    rpt_state_key: str,
    active_only: bool,
) -> gpd.GeoDataFrame:
    if not uwwtd_agglomerations_filepath.is_file():
        log.warning(f"UWWTD agglomerations file missing: {uwwtd_agglomerations_filepath}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.read_file(uwwtd_agglomerations_filepath, datetime_as_string=True)
    if gdf.empty:
        return gdf

    key = str(rpt_state_key).strip().upper()
    if "rptMStateKey" not in gdf.columns:
        log.warning("UWWTD agglomerations GPKG has no rptMStateKey column")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    rk = gdf["rptMStateKey"].astype(str).str.strip().str.upper()
    gdf = gdf[rk.eq(key)].copy()
    if gdf.empty:
        log.info(f"UWWTD agglomerations: no rows for rptMStateKey={key!r}")
        return gdf

    if active_only and "aggState" in gdf.columns:
        st = pd.to_numeric(gdf["aggState"], errors="coerce")
        gdf = gdf[st.fillna(0).eq(1)]

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _rasterize_buffered_geoms(
    gdf: gpd.GeoDataFrame,
    *,
    buffer_m: float,
    metric_crs: str,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    rasterize_cfg: dict[str, Any],
) -> np.ndarray:
    """Buffer geometries in *metric_crs*, then rasterize onto the CORINE reference grid."""
    fill = float(rasterize_cfg["fill"])
    burn = float(rasterize_cfg["burn_value"])
    dtype = np.dtype(rasterize_cfg["dtype"])
    all_touched = bool(rasterize_cfg["all_touched"])

    acc = np.full((height, width), fill, dtype=np.float64)
    if gdf is None or gdf.empty:
        return acc.astype(dtype, copy=False)

    gx = gdf.to_crs(metric_crs)
    buf = float(buffer_m)
    if buf > 0:
        gx = gx.copy()
        gx["geometry"] = gx.geometry.buffer(buf)
    gx = gx.to_crs(raster_crs)

    shapes = ((geom, burn) for geom in gx.geometry if geom is not None and not geom.is_empty)
    rio_features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        out=acc,
        dtype=np.float64,
        all_touched=all_touched,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    inside = pixels_inside_cams_cells(height, width, transform, raster_crs, cams_cells)
    acc = np.where(inside, acc, fill)
    return acc.astype(dtype, copy=False)


def load_uwwtd_agglomerations(
    uwwtd_agglomerations_filepath: Path,
    *,
    rpt_state_key: str,
    metric_crs: str,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    buffer_m: float = 50.0,
    active_only: bool = True,
    rasterize_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    UWWTD agglomeration polygons (or points) buffered by *buffer_m* in *metric_crs*, then rasterized
    to the reference grid (same contract as ``rasterize_osm``).
    """
    cfg = rasterize_cfg or {
        "burn_value": 1.0,
        "fill": 0.0,
        "dtype": "float32",
        "all_touched": True,
    }
    gdf = _read_uwwtd_agg_gdf(
        uwwtd_agglomerations_filepath,
        rpt_state_key=rpt_state_key,
        active_only=active_only,
    )
    out = _rasterize_buffered_geoms(
        gdf,
        buffer_m=buffer_m,
        metric_crs=metric_crs,
        height=height,
        width=width,
        transform=transform,
        raster_crs=raster_crs,
        cams_cells=cams_cells,
        rasterize_cfg=cfg,
    )
    log.info(
        f"UWWTD agglomerations raster: buffer_m={buffer_m:g} max={float(np.nanmax(out)):.4f} "
        f"({int(np.count_nonzero(out))} non-zero px)"
    )
    return out


def load_uwwtd_treatment_plants_raster(
    uwwtd_treatment_plants_filepath: Path,
    *,
    rpt_state_key: str,
    metric_crs: str,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    buffer_m: float = 500.0,
    active_only: bool = True,
    rasterize_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    UWWTD treatment plant geometries buffered by *buffer_m* (default 500 m) in *metric_crs*,
    then rasterized on the CORINE reference grid.
    """
    cfg = rasterize_cfg or {
        "burn_value": 1.0,
        "fill": 0.0,
        "dtype": "float32",
        "all_touched": True,
    }
    gdf = _read_uwwtd_plants_gdf(
        uwwtd_treatment_plants_filepath,
        rpt_state_key=rpt_state_key,
        active_only=active_only,
    )
    out = _rasterize_buffered_geoms(
        gdf,
        buffer_m=buffer_m,
        metric_crs=metric_crs,
        height=height,
        width=width,
        transform=transform,
        raster_crs=raster_crs,
        cams_cells=cams_cells,
        rasterize_cfg=cfg,
    )
    log.info(
        f"UWWTD treatment plants raster: buffer_m={buffer_m:g} max={float(np.nanmax(out)):.4f} "
        f"({int(np.count_nonzero(out))} non-zero px)"
    )
    return out


def load_uwwtd_treatment_plants(
    uwwtd_treatment_plants_filepath: Path,
    *,
    rpt_state_key: str,
    active_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Point-matching dict: same rows as the raster path, with ``lon`` / ``lat`` per facility.

    *rpt_state_key* is normally ``country_profile['other']`` (e.g. ``EL`` for Greece).
    """
    gdf = _read_uwwtd_plants_gdf(
        uwwtd_treatment_plants_filepath,
        rpt_state_key=rpt_state_key,
        active_only=active_only,
    )
    if gdf.empty:
        return {}

    key = str(rpt_state_key).strip().upper()
    out: dict[str, dict[str, Any]] = {}
    for i, (_, row) in enumerate(gdf.iterrows()):
        uid = _row_uid(row, f"UWWTD_{i}")
        if uid in out:
            uid = f"{uid}_{i}"

        lon_s = pd.to_numeric(row.get("uwwLongitude"), errors="coerce")
        lat_s = pd.to_numeric(row.get("uwwLatitude"), errors="coerce")
        geom = row.geometry
        if pd.notna(lon_s) and pd.notna(lat_s):
            lon, lat = float(lon_s), float(lat_s)
        elif geom is not None and not geom.is_empty:
            c = geom.centroid
            lon, lat = float(c.x), float(c.y)
        else:
            continue

        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue

        name = str(row.get("uwwName") or "").strip()
        out[uid] = {
            "facility_id": uid,
            "facility_name": name or uid,
            "lon": lon,
            "lat": lat,
            "uww_code": str(row.get("uwwCode") or "").strip(),
            "nuts": str(row.get("uwwNUTS") or "").strip(),
        }

    log.info(f"UWWTD treatment plants (points) for rptMStateKey={key!r}: {len(out)} facilities")
    return out
