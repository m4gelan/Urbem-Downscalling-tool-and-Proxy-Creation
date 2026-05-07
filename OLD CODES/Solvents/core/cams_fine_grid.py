"""Map each fine-grid pixel to a CAMS GNFR E *area* source index (parent cell)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as rio_transform_pts


IDX_E = 5
IDX_AREA = 1


def _decode_country_ids(ds: xr.Dataset) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for v in raw:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", "replace").strip())
        else:
            out.append(str(v).strip())
    return out


def country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    return codes.index(u) + 1


def gnfr_e_area_mask(ds: xr.Dataset, iso3: str) -> np.ndarray:
    ix = country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_E) & (st == IDX_AREA) & (ci == ix)
    return base


def cams_indices(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
        lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
        lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
    lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
    lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)
    return lon_ii, lat_ii, lon_b, lat_b


def build_cell_of(
    nc_path: Path,
    iso3: str,
    ref: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (cell_of, m_e_mask).

    cell_of: (H,W) int64, CAMS source index for GNFR E area parent, or -1.
    m_e_mask: 1D bool over CAMS sources — True for GNFR E area rows in this country.
    """
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    ds = xr.open_dataset(nc_path)
    try:
        lon_ii, lat_ii, lon_b, lat_b = cams_indices(ds)
        m_e = gnfr_e_area_mask(ds, iso3)

        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lookup = np.full(nlon * nlat, -1, dtype=np.int64)
        for i in sorted(int(x) for x in np.flatnonzero(m_e)):
            li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
            k = li_i * nlat + ji_i
            if lookup[k] < 0:
                lookup[k] = int(i)

        cell_of = np.empty((h, w), dtype=np.int64)
        row_chunk = 512
        cols_idx = np.arange(w, dtype=np.float64)
        for r0 in range(0, h, row_chunk):
            r1 = min(h, r0 + row_chunk)
            rh = r1 - r0
            rr = np.broadcast_to(
                np.arange(r0, r1, dtype=np.float64)[:, None],
                (rh, w),
            )
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
            cell_of[r0:r1, :] = np.where(
                in_bounds & (cid >= 0), cid, -1
            ).astype(np.int64)
        return cell_of, m_e
    finally:
        ds.close()
