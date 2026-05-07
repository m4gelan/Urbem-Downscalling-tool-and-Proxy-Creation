"""
Map each fine-grid pixel to a CAMS native lon/lat **grid cell** id (not NetCDF source index).

Uses ``longitude_bounds`` / ``latitude_bounds`` from CAMS-REG-ANT v8.1 and pixel-center
coordinates transformed to WGS84, matching the approach in ``Solvents/core/cams_fine_grid.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as rio_transform_pts

logger = logging.getLogger(__name__)


def read_cams_bounds(nc_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (lon_b, lat_b, nlon, nlat) from NetCDF."""
    ds = xr.open_dataset(nc_path)
    try:
        lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        return lon_b, lat_b, nlon, nlat
    finally:
        ds.close()


def build_cam_cell_id(nc_path: Path, ref: dict) -> np.ndarray:
    """
    For each fine pixel (center), compute CAMS grid indices ``li`` (lon), ``ji`` (lat).

    Returns ``cam_cell_id`` int32 array (H, W):
      - ``ji * nlon + li`` when the pixel center lies inside the CAMS cell rectangle
      - ``-1`` when outside all cells (e.g. outside CAMS domain) or invalid

    This id is used only for **grouping** fine pixels that fall in the same CAMS raster cell.
    It is **not** the ``source`` dimension index from the NetCDF.
    """
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])
    lon_b, lat_b, nlon, nlat = read_cams_bounds(nc_path)

    out = np.full((h, w), -1, dtype=np.int32)
    row_chunk = 512
    cols_idx = np.arange(w, dtype=np.float64)
    for r0 in range(0, h, row_chunk):
        r1 = min(h, r0 + row_chunk)
        rh = r1 - r0
        rr = np.broadcast_to(np.arange(r0, r1, dtype=np.float64)[:, None], (rh, w))
        cc = np.broadcast_to(cols_idx, (rh, w))
        xs, ys = transform_xy(transform, rr + 0.5, cc + 0.5, offset="center")
        lons, lats = rio_transform_pts(crs, "EPSG:4326", xs.ravel(), ys.ravel())
        lons = np.asarray(lons, dtype=np.float64).reshape(rh, w)
        lats = np.asarray(lats, dtype=np.float64).reshape(rh, w)
        li = np.searchsorted(lon_b[:, 0], lons, side="right") - 1
        li = np.clip(li, 0, nlon - 1)
        ji = np.searchsorted(lat_b[:, 0], lats, side="right") - 1
        ji = np.clip(ji, 0, nlat - 1)
        valid_lon = (lons >= lon_b[li, 0]) & (lons <= lon_b[li, 1])
        valid_lat = (lats >= lat_b[ji, 0]) & (lats <= lat_b[ji, 1])
        in_bounds = valid_lon & valid_lat
        cid = (ji * nlon + li).astype(np.int32)
        out[r0:r1, :] = np.where(in_bounds, cid, -1)
    n_valid = int(np.count_nonzero(out >= 0))
    logger.info(
        "cam_cell_id: %s valid pixels (%.1f%% of grid) on CAMS domain %s x %s cells",
        n_valid,
        100.0 * n_valid / max(h * w, 1),
        nlon,
        nlat,
    )
    return out


def cams_grid_shape(nc_path: Path) -> tuple[int, int]:
    """Return (nlon, nlat) for diagnostics."""
    lon_b, lat_b, nlon, nlat = read_cams_bounds(nc_path)
    return nlon, nlat


def j_area_source_mask(nc_path: Path, gnfr_j_index: int = 13) -> np.ndarray:
    """
    1D bool mask over NetCDF ``source`` dimension: GNFR J and area sources.
    Useful for optional diagnostics (which CAMS cells carry J area records).
    """
    ds = xr.open_dataset(nc_path)
    try:
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        return (emis == int(gnfr_j_index)) & (st == 1)
    finally:
        ds.close()


def _country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    raw = ds["country_id"].values
    codes: list[str] = []
    for v in raw:
        if isinstance(v, bytes):
            codes.append(v.decode("utf-8", "replace").strip())
        else:
            codes.append(str(v).strip())
    return codes.index(iso3.strip().upper()) + 1


def j_geographic_cids_for_sources(
    nc_path: Path,
    *,
    gnfr_j_index: int,
    source_type: int,
    country_iso3: str,
) -> set[int]:
    """
    Geographic fine-grid cell ids ``ji * nlon + li`` (same encoding as ``build_cam_cell_id``)
    that contain at least one CAMS source with GNFR J, ``country_iso3``, and ``source_type``
    (CAMS-REG: 1 = area, 2 = point).
    """
    _, _, nlon, nlat = read_cams_bounds(nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        try:
            ix = _country_index_1based(ds, country_iso3)
        except ValueError:
            logger.warning(
                "CAMS country_id has no entry for %s; GNFR J geographic masks empty.",
                country_iso3,
            )
            return set()
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
        lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
        if lon_idx_raw.size > 0 and (
            lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat
        ):
            lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
            lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
        li = np.clip(lon_idx_raw, 0, nlon - 1)
        ji = np.clip(lat_idx_raw, 0, nlat - 1)
        m = (emis == int(gnfr_j_index)) & (st == int(source_type)) & (ci == ix)
        if not np.any(m):
            return set()
        out: set[int] = set()
        for i in np.flatnonzero(m):
            out.add(int(ji[i] * nlon + li[i]))
        return out
    finally:
        ds.close()


def build_cam_cell_id_masked_for_j_sources(
    nc_path: Path,
    ref: dict,
    *,
    gnfr_j_index: int,
    source_type: int,
    country_iso3: str,
) -> np.ndarray:
    """
    Like ``build_cam_cell_id`` but pixels are kept only when their geographic CAMS cell
    contains at least one GNFR J source of the given ``source_type`` for ``country_iso3``.
    Other pixels are ``-1``.
    """
    base = build_cam_cell_id(nc_path, ref)
    valid = j_geographic_cids_for_sources(
        nc_path,
        gnfr_j_index=gnfr_j_index,
        source_type=source_type,
        country_iso3=country_iso3,
    )
    if not valid:
        return np.full_like(base, -1)
    arr = np.array(sorted(valid), dtype=np.int32)
    return np.where(np.isin(base, arr), base, -1).astype(np.int32)
