from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from UrbEm_Visualizer.dataset_loaders.cams_alias import (
    cams_country_index_from_iso3,
    cams_pollutant_var,
)


def cell_id_from_lonlat(
    flat_lon: np.ndarray,
    flat_lat: np.ndarray,
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
    nlon: int,
    nlat: int,
) -> np.ndarray:
    lo = np.searchsorted(lon_bounds[:, 0], flat_lon, side="right") - 1
    lo = np.clip(lo, 0, nlon - 1)
    ok_lon = (flat_lon >= lon_bounds[lo, 0]) & (flat_lon < lon_bounds[lo, 1])
    la = np.searchsorted(lat_bounds[:, 0], flat_lat, side="right") - 1
    la = np.clip(la, 0, nlat - 1)
    ok_lat = (flat_lat >= lat_bounds[la, 0]) & (flat_lat < lat_bounds[la, 1])
    cid = la.astype(np.int32) * int(nlon) + lo.astype(np.int32)
    cid[~(ok_lon & ok_lat)] = -1
    return cid


def load_cams_area_cells(
    cams_nc: Path,
    *,
    year: int,
    country_iso3: str,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    iso3 = str(country_iso3).strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_filter = np.asarray(source_type_indices, dtype=np.int64)
    labels = [p.strip() for p in pollutants if p.strip()]
    if not labels:
        raise ValueError("pollutants must be non-empty")

    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        country_idx = cams_country_index_from_iso3(ds, iso3)
        lon_src = np.asarray(ds["longitude_source"].values, dtype=np.float64).ravel()
        lat_src = np.asarray(ds["latitude_source"].values, dtype=np.float64).ravel()
        src_type = np.asarray(ds["source_type_index"].values, dtype=np.int64).ravel()
        emis_cat = np.asarray(ds["emission_category_index"].values, dtype=np.int64).ravel()
        country_index = np.asarray(ds["country_index"].values, dtype=np.int64).ravel()
        lon_idx = np.asarray(ds["longitude_index"].values, dtype=np.int64).ravel()
        lat_idx = np.asarray(ds["latitude_index"].values, dtype=np.int64).ravel()
        nlon = int(ds.sizes["longitude"])
        nlat = int(ds.sizes["latitude"])
        lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        if lon_idx.size and (lon_idx.max() >= nlon or lat_idx.max() >= nlat):
            lon_idx = lon_idx - 1
            lat_idx = lat_idx - 1
        np.clip(lon_idx, 0, nlon - 1, out=lon_idx)
        np.clip(lat_idx, 0, nlat - 1, out=lat_idx)
        pol_mat = np.column_stack([
            np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float32).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for lab in labels
        ])

    mask = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & np.isin(emis_cat, ec_filter)
        & np.isin(src_type, st_filter)
        & (country_index == int(country_idx))
        & (pol_mat.max(axis=1) > 0.0)
    )
    sel = np.flatnonzero(mask)
    grid = {
        "lon_bounds": lon_bounds,
        "lat_bounds": lat_bounds,
        "n_longitude": nlon,
        "n_latitude": nlat,
    }
    if sel.size == 0:
        return {}, grid

    cell_ids = lat_idx[sel] * nlon + lon_idx[sel]
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    n_cells = unique_cells.size
    sums = np.empty((n_cells, len(labels)), dtype=np.float32)
    for j in range(len(labels)):
        sums[:, j] = np.bincount(inverse, weights=pol_mat[sel, j], minlength=n_cells).astype(np.float32)

    out: dict[int, dict[str, Any]] = {}
    for k, cell_id in enumerate(unique_cells.tolist()):
        lo = int(cell_id % nlon)
        la = int(cell_id // nlon)
        west, east = float(lon_bounds[lo, 0]), float(lon_bounds[lo, 1])
        south, north = float(lat_bounds[la, 0]), float(lat_bounds[la, 1])
        out[int(cell_id)] = {
            "pollutants_within_cell": {lab: float(sums[k, j]) for j, lab in enumerate(labels)},
            "cell_bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
        }
    return out, grid


def load_cams_points(
    cams_nc: Path,
    *,
    year: int,
    country_iso3: str,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
) -> dict[int, dict[str, Any]]:
    iso3 = str(country_iso3).strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_filter = np.asarray(source_type_indices, dtype=np.int64)
    labels = [p.strip() for p in pollutants if p.strip()]

    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        country_idx = cams_country_index_from_iso3(ds, iso3)
        lon = np.asarray(ds["longitude_source"].values, dtype=np.float64).ravel()
        lat = np.asarray(ds["latitude_source"].values, dtype=np.float64).ravel()
        src_type = np.asarray(ds["source_type_index"].values, dtype=np.int64).ravel()
        emis_cat = np.asarray(ds["emission_category_index"].values, dtype=np.int64).ravel()
        country_index = np.asarray(ds["country_index"].values, dtype=np.int64).ravel()
        pol_m = {
            lab: np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float32).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for lab in labels
        }

    mask = np.isfinite(lon) & np.isfinite(lat)
    mask &= np.isin(emis_cat, ec_filter)
    mask &= np.isin(src_type, st_filter)
    mask &= country_index == int(country_idx)
    if pol_m:
        mask &= np.max(np.stack(list(pol_m.values()), axis=1), axis=1) > 0.0

    out: dict[int, dict[str, Any]] = {}
    for i in np.flatnonzero(mask):
        out[int(i)] = {
            "latitude": float(lat[i]),
            "longitude": float(lon[i]),
            "pollutants": {lab: float(pol_m[lab][i]) for lab in labels},
            "year": int(year),
        }
    return out


def load_cams_grid_meta(cams_nc: Path) -> dict[str, Any]:
    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        return {
            "lon_bounds": np.asarray(ds["longitude_bounds"].values, dtype=np.float64),
            "lat_bounds": np.asarray(ds["latitude_bounds"].values, dtype=np.float64),
            "n_longitude": int(ds.sizes["longitude"]),
            "n_latitude": int(ds.sizes["latitude"]),
        }


def point_cell_ids(
    cams_points: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
) -> frozenset[int]:
    lon_bounds = cams_grid["lon_bounds"]
    lat_bounds = cams_grid["lat_bounds"]
    nlon = int(cams_grid["n_longitude"])
    nlat = int(cams_grid["n_latitude"])
    ids = []
    for row in cams_points.values():
        cid = cell_id_from_lonlat(
            np.array([row["longitude"]], dtype=np.float64),
            np.array([row["latitude"]], dtype=np.float64),
            lon_bounds,
            lat_bounds,
            nlon,
            nlat,
        )
        if cid[0] >= 0:
            ids.append(int(cid[0]))
    return frozenset(ids)


def cams_mass_lookup(cells: dict[int, dict[str, Any]], pollutants: list[str]) -> dict[str, np.ndarray]:
    if not cells:
        return {p: np.zeros(1, dtype=np.float32) for p in pollutants}
    max_id = max(cells)
    out = {}
    for pol in pollutants:
        arr = np.zeros(max_id + 1, dtype=np.float32)
        for cid, row in cells.items():
            arr[cid] = np.float32((row.get("pollutants_within_cell") or {}).get(pol, 0.0))
        out[pol] = arr
    return out
