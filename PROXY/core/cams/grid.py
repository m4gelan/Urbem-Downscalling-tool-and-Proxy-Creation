"""Single source of truth for fine-grid -> CAMS geographic cell id mapping.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as rio_transform_pts

from PROXY.core.cams.domain import country_index_1based

logger = logging.getLogger(__name__)


def read_cams_bounds(nc_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
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
    """Assign to each fine-grid pixel the linear CAMS cell id ``ji * nlon + li``.

    Pixels whose center falls outside any CAMS cell box get ``-1``.
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


def geographic_cids_for_sources(
    nc_path: Path,
    *,
    gnfr_index: int,
    source_type_index: int,
    country_iso3: str,
) -> set[int]:
    """Return CAMS geographic cell ids that have matching source rows."""
    _, _, nlon, nlat = read_cams_bounds(nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        try:
            src_mask = cams_source_mask(
                ds,
                gnfr_index=gnfr_index,
                source_type_index=source_type_index,
                country_iso3=country_iso3,
            )
        except ValueError:
            logger.warning(
                "CAMS country_id has no entry for %s; source geographic masks empty.",
                country_iso3,
            )
            return set()

        lon_idx, lat_idx, _, _ = _source_lon_lat_indices(ds)
        if not np.any(src_mask):
            return set()
        out: set[int] = set()
        for i in np.flatnonzero(src_mask):
            out.add(int(int(lat_idx[i]) * nlon + int(lon_idx[i])))
        return out
    finally:
        ds.close()


def build_cam_cell_id_masked_for_sources(
    nc_path: Path,
    ref: dict,
    *,
    gnfr_index: int,
    source_type_index: int,
    country_iso3: str,
) -> np.ndarray:
    """Build a fine-grid CAMS cell-id raster, masked to matching source rows."""
    base = build_cam_cell_id(nc_path, ref)
    valid = geographic_cids_for_sources(
        nc_path,
        gnfr_index=gnfr_index,
        source_type_index=source_type_index,
        country_iso3=country_iso3,
    )
    if not valid:
        return np.full_like(base, -1)
    arr = np.array(sorted(valid), dtype=np.int32)
    return np.where(np.isin(base, arr), base, -1).astype(np.int32)


def _source_lon_lat_indices(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx.max() >= nlon or lat_idx.max() >= nlat:
        lon_idx = np.maximum(0, lon_idx - 1)
        lat_idx = np.maximum(0, lat_idx - 1)
    lon_idx = np.clip(lon_idx, 0, nlon - 1)
    lat_idx = np.clip(lat_idx, 0, nlat - 1)
    return lon_idx, lat_idx, lon_b, lat_b


def cams_source_mask(
    ds: xr.Dataset,
    *,
    gnfr_index: int,
    source_type_index: int,
    country_iso3: str,
) -> np.ndarray:
    ix = country_index_1based(ds, country_iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    return (emis == int(gnfr_index)) & (st == int(source_type_index)) & (ci == ix)


def cams_source_mask_any_gnfr(
    ds: xr.Dataset,
    *,
    gnfr_indices: list[int] | tuple[int, ...],
    source_type_index: int,
    country_iso3: str,
) -> np.ndarray:
    """Boolean CAMS source-row mask for any GNFR index in ``gnfr_indices``."""
    ix = country_index_1based(ds, country_iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    gnfr_arr = np.asarray([int(x) for x in gnfr_indices], dtype=np.int64)
    return np.isin(emis, gnfr_arr) & (st == int(source_type_index)) & (ci == ix)


def build_cams_source_index_grid(
    nc_path: Path,
    ref: dict,
    *,
    gnfr_index: int,
    source_type_index: int,
    country_iso3: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map fine pixels to the CAMS NetCDF source-row index matching sector/type/country.

    The returned raster stores the original NetCDF source row id, not just a geographic
    cell id. This is required when later normalization must preserve CAMS source rows.
    """
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    ds = xr.open_dataset(nc_path)
    try:
        lon_ii, lat_ii, lon_b, lat_b = _source_lon_lat_indices(ds)
        src_mask = cams_source_mask(
            ds,
            gnfr_index=gnfr_index,
            source_type_index=source_type_index,
            country_iso3=country_iso3,
        )

        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lookup = np.full(nlon * nlat, -1, dtype=np.int64)
        for i in sorted(int(x) for x in np.flatnonzero(src_mask)):
            li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
            k = li_i * nlat + ji_i
            if lookup[k] < 0:
                lookup[k] = int(i)

        out = np.empty((h, w), dtype=np.int64)
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
            fk = li * nlat + ji
            cid = lookup[fk]
            out[r0:r1, :] = np.where(in_bounds & (cid >= 0), cid, -1).astype(np.int64)
        return out, src_mask
    finally:
        ds.close()


def build_cams_source_index_grid_any_gnfr(
    nc_path: Path,
    ref: dict,
    *,
    gnfr_indices: list[int] | tuple[int, ...],
    source_type_index: int,
    country_iso3: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map fine pixels to CAMS source rows matching any configured GNFR category."""
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    ds = xr.open_dataset(nc_path)
    try:
        lon_ii, lat_ii, lon_b, lat_b = _source_lon_lat_indices(ds)
        src_mask = cams_source_mask_any_gnfr(
            ds,
            gnfr_indices=gnfr_indices,
            source_type_index=source_type_index,
            country_iso3=country_iso3,
        )

        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lookup = np.full(nlon * nlat, -1, dtype=np.int64)
        for i in sorted(int(x) for x in np.flatnonzero(src_mask)):
            li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
            k = li_i * nlat + ji_i
            if lookup[k] < 0:
                lookup[k] = int(i)

        out = np.empty((h, w), dtype=np.int64)
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
            fk = li * nlat + ji
            cid = lookup[fk]
            out[r0:r1, :] = np.where(in_bounds & (cid >= 0), cid, -1).astype(np.int64)
        return out, src_mask
    finally:
        ds.close()
