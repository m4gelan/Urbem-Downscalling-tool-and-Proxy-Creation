"""CAMS cell-id rasters for sector allocation workflows.

Inputs are a CAMS NetCDF path, a reference grid profile, GNFR/source-type
indices, and a country ISO3 code. The output is a raster of CAMS geographic
cell ids, masked to cells that have matching CAMS source rows. This preserves
the legacy `area_allocation.build_cams_cell_id_raster` behavior while placing
the CAMS-specific implementation under `core.cams`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from PROXY.core.cams.domain import country_index_1based


def build_cams_cell_id_raster(
    *,
    cams_nc_path: Path,
    ref_profile: dict[str, Any],
    gnfr_index: int,
    source_type_index: int,
    country_iso3: str,
) -> np.ndarray:
    """Build a CAMS geographic cell-id raster masked by GNFR, source type, and country."""
    with xr.open_dataset(cams_nc_path, engine="netcdf4") as ds:
        lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        lon_index = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
        lat_index = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        cidx = country_index_1based(ds, country_iso3)

    nlon = lon_bounds.shape[0]
    nlat = lat_bounds.shape[0]
    if lon_index.max() >= nlon:
        lon_index = np.maximum(lon_index - 1, 0)
    if lat_index.max() >= nlat:
        lat_index = np.maximum(lat_index - 1, 0)
    lon_index = np.clip(lon_index, 0, nlon - 1)
    lat_index = np.clip(lat_index, 0, nlat - 1)

    src_mask = (emis == int(gnfr_index)) & (st == int(source_type_index)) & (ci == int(cidx))
    valid_cells = set(
        zip(lat_index[src_mask].tolist(), lon_index[src_mask].tolist(), strict=False)
    )
    if not valid_cells:
        return np.full((int(ref_profile["height"]), int(ref_profile["width"])), -1, dtype=np.int64)

    t = ref_profile["transform"]
    h = int(ref_profile["height"])
    w = int(ref_profile["width"])
    xs = np.array([t.c + (c + 0.5) * t.a for c in range(w)], dtype=np.float64)
    ys = np.array([t.f + (r + 0.5) * t.e for r in range(h)], dtype=np.float64)

    lon_min = np.minimum(lon_bounds[:, 0], lon_bounds[:, 1])
    lon_max = np.maximum(lon_bounds[:, 0], lon_bounds[:, 1])
    lat_min = np.minimum(lat_bounds[:, 0], lat_bounds[:, 1])
    lat_max = np.maximum(lat_bounds[:, 0], lat_bounds[:, 1])

    lon_idx = np.searchsorted(lon_max, xs, side="right")
    lon_ok = (lon_idx >= 0) & (lon_idx < nlon) & (xs >= lon_min[np.clip(lon_idx, 0, nlon - 1)])

    lat_idx = np.searchsorted(lat_max, ys, side="right")
    lat_ok = (lat_idx >= 0) & (lat_idx < nlat) & (ys >= lat_min[np.clip(lat_idx, 0, nlat - 1)])

    out = np.full((h, w), -1, dtype=np.int64)
    for r in range(h):
        if not lat_ok[r]:
            continue
        li = int(lat_idx[r])
        row_lon = lon_idx[lon_ok]
        cols = np.where(lon_ok)[0]
        for c, lj in zip(cols.tolist(), row_lon.tolist(), strict=False):
            if (li, int(lj)) in valid_cells:
                out[r, c] = int(li) * nlon + int(lj)
    return out


__all__ = ["build_cams_cell_id_raster"]
