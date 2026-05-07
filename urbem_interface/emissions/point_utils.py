"""
Utilities for the point-sources pipeline.

Kept separate from `point_sources.py` to keep the pipeline orchestration readable,
similar to how grid warping lives in `grid_functions/grid_warp.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS

from urbem_interface.utils.grid import _compute_proxy_coarse_grid
from urbem_interface.logging_config import get_logger
from urbem_interface.utils.domain import domain_bounds_wgs84

logger = get_logger(__name__)


def read_points_config(path: Path) -> dict[str, Any]:
    """Load point-sources JSON config from *path*."""
    import json

    logger.debug(f"Reading point-sources config: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    logger.debug(f"  -> {len(cfg)} top-level keys: {list(cfg.keys())}")
    return cfg


def coarse_grid_from_cams(
    cams_transform_wgs: rasterio.Affine,
    nlat: int,
    nlon: int,
    domain_crs: CRS,
    domain_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, Affine, int, int, tuple[float, float, float, float]]:
    """
    Build the coarse CAMS grid for point-source normalisation.

    Reuses the same logic as area sources:
    projectRaster() to domain_crs at 7000 x 5550 m, then snap-out crop.

    Returns
    -------
    cams_origin_arr  : reprojected CAMS id raster
    coarse_transform : Affine transform of the coarse grid (domain CRS)
    nlon_c, nlat_c   : coarse grid dimensions
    coarse_bounds    : (xmin, ymin, xmax, ymax) in domain CRS
    """
    wgs84 = CRS.from_epsg(4326)
    logger.debug(
        f"Building coarse CAMS grid  src_shape=({nlat}, {nlon})  "
        f"res=7000x5550 m  domain_bounds={domain_bounds}"
    )
    result = _compute_proxy_coarse_grid(
        src_shape=(nlat, nlon),
        src_transform=cams_transform_wgs,
        src_crs=wgs84,
        dst_crs=domain_crs,
        domain_bounds=domain_bounds,
        res_x=7000.0,
        res_y=5550.0,
    )
    _, coarse_transform, nlon_c, nlat_c, coarse_bounds = result
    logger.debug(
        f"  -> coarse grid: {nlat_c} rows x {nlon_c} cols  "
        f"bounds={coarse_bounds}"
    )
    return result


def assign_cams_cell_indices(
    points_df: pd.DataFrame,
    lons: np.ndarray,
    lats: np.ndarray,
) -> pd.DataFrame:
    """
    Assign each point to a CAMS (lat_idx, lon_idx) using regular-grid bin edges.

    Matches R intent: cell extent = (center +/- res/2) in each axis.
    Handles both north-down (lats descending) and south-up (lats ascending) grids.

    Parameters
    ----------
    points_df : must contain columns "lon" and "lat" in WGS84.
    lons      : 1-D array of CAMS cell-centre longitudes.
    lats      : 1-D array of CAMS cell-centre latitudes.

    Returns
    -------
    Copy of points_df with added integer columns "lon_idx" and "lat_idx".
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if lons.size < 2 or lats.size < 2:
        raise ValueError(
            f"CAMS grid lons/lats must have >=2 values to compute resolution. "
            f"Got lons={lons.size}, lats={lats.size}."
        )

    res_lon = float(abs(lons[1] - lons[0]))
    res_lat = float(abs(lats[1] - lats[0]))
    logger.debug(
        f"CAMS grid resolution: delta lon={res_lon:.4f} deg  delta lat={res_lat:.4f} deg  "
        f"lon=[{lons[0]:.3f}...{lons[-1]:.3f}]  lat=[{lats[0]:.3f}...{lats[-1]:.3f}]"
    )

    lon_edges = np.concatenate([[lons[0] - res_lon / 2], lons + res_lon / 2])

    lon_vals = points_df["lon"].to_numpy(float)
    lat_vals = points_df["lat"].to_numpy(float)

    if lats[0] > lats[-1]:
        # North-down: descending latitude array
        lat_edges_desc = np.concatenate([[lats[0] + res_lat / 2], lats - res_lat / 2])
        lat_edges = lat_edges_desc[::-1]
        lat_bin_from_south = np.searchsorted(lat_edges, lat_vals, side="right") - 1
        lat_idx = (lats.size - 1) - lat_bin_from_south
    else:
        # South-up: ascending latitude array
        lat_edges = np.concatenate([[lats[0] - res_lat / 2], lats + res_lat / 2])
        lat_idx = np.searchsorted(lat_edges, lat_vals, side="right") - 1

    lon_idx = np.searchsorted(lon_edges, lon_vals, side="right") - 1

    # Diagnostics
    n_total = len(points_df)
    n_out_lon = int(np.sum((lon_idx < 0) | (lon_idx >= lons.size)))
    n_out_lat = int(np.sum((lat_idx < 0) | (lat_idx >= lats.size)))
    if n_out_lon or n_out_lat:
        logger.warning(
            f"assign_cams_cell_indices: {n_out_lon}/{n_total} points outside lon range, "
            f"{n_out_lat}/{n_total} outside lat range - these will produce invalid indices"
        )
    else:
        logger.debug(f"  -> all {n_total} points successfully assigned to CAMS cells")

    out = points_df.copy()
    out["lon_idx"] = lon_idx.astype(int)
    out["lat_idx"] = lat_idx.astype(int)
    return out


def project_points_to_domain(
    points_df: pd.DataFrame,
    domain_crs: CRS,
) -> pd.DataFrame:
    """
    Reproject point coordinates from WGS84 to domain CRS.

    Expects "lon" and "lat" columns in points_df.
    Adds "xcor" and "ycor" columns (domain CRS metres).
    """
    to_dom = Transformer.from_crs(CRS.from_epsg(4326), domain_crs, always_xy=True)

    lon_arr = points_df["lon"].to_numpy(float)
    lat_arr = points_df["lat"].to_numpy(float)
    xs, ys  = to_dom.transform(lon_arr, lat_arr)

    logger.debug(
        f"Projected {len(points_df):,} points to domain CRS  "
        f"x=[{xs.min():.0f}, {xs.max():.0f}]  "
        f"y=[{ys.min():.0f}, {ys.max():.0f}]"
    )

    out = points_df.copy()
    out["xcor"] = xs
    out["ycor"] = ys
    return out


def rasterize_unmatched_to_coarse(
    unmatched: pd.DataFrame,
    coarse_transform: Affine,
    coarse_shape: tuple[int, int],
) -> np.ndarray:
    """
    Accumulate unmatched point emissions into coarse-grid cells.

    Points that fall outside the coarse grid extent are silently dropped
    (they were already unmatched and have no valid CAMS cell to fall back on).

    Parameters
    ----------
    unmatched        : DataFrame with "xcor", "ycor", "emission" columns (domain CRS).
    coarse_transform : Affine transform of the coarse grid.
    coarse_shape     : (nrow, ncol) of the coarse grid.

    Returns
    -------
    2-D float64 array of shape coarse_shape with summed emissions per cell.
    """
    nrow, ncol = coarse_shape
    res_x = float(coarse_transform.a)
    res_y = float(abs(coarse_transform.e))
    left  = float(coarse_transform.c)
    top   = float(coarse_transform.f)

    x_arr = unmatched["xcor"].to_numpy(float)
    y_arr = unmatched["ycor"].to_numpy(float)
    v_arr = unmatched["emission"].to_numpy(float)

    col = np.floor((x_arr - left) / res_x).astype(int)
    row = np.floor((top  - y_arr) / res_y).astype(int)

    valid = (row >= 0) & (row < nrow) & (col >= 0) & (col < ncol)
    n_dropped = int((~valid).sum())
    n_kept    = int(valid.sum())

    if n_dropped:
        logger.warning(
            f"rasterize_unmatched_to_coarse: {n_dropped}/{len(unmatched)} points "
            f"fall outside coarse grid extent and will be dropped"
        )

    row   = row[valid]
    col   = col[valid]
    v_arr = v_arr[valid]

    out = np.zeros((nrow, ncol), dtype=np.float64)
    if v_arr.size:
        np.add.at(out, (row, col), v_arr)

    logger.debug(
        f"rasterize_unmatched_to_coarse: {n_kept} points rasterized  "
        f"total_emission={out.sum():.4e}  "
        f"non_zero_cells={int((out > 0).sum())}/{nrow * ncol}"
    )
    return out
