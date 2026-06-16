from __future__ import annotations

import numpy as np


def haversine_km(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in km between (lon1, lat1) and (lon2, lat2)."""
    lon1r = np.radians(np.asarray(lon1, dtype=np.float64))
    lat1r = np.radians(np.asarray(lat1, dtype=np.float64))
    lon2r = np.radians(np.asarray(lon2, dtype=np.float64))
    lat2r = np.radians(np.asarray(lat2, dtype=np.float64))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def lon_lat_to_cell_ids(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    longitude_bounds: np.ndarray,
    latitude_bounds: np.ndarray,
    n_longitude: int,
    n_latitude: int,
) -> np.ndarray:
    """Map WGS84 (lon, lat) to CAMS ``cell_id``; ``-1`` where outside the grid.

    Uses the same half-open bounds as :func:`pixels_inside_cams_cells`:
    ``west <= lon < east``, ``south <= lat < north``.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lon_w = longitude_bounds[:, 0]
    lon_e = longitude_bounds[:, 1]
    lat_s = latitude_bounds[:, 0]
    lat_n = latitude_bounds[:, 1]

    in_lon = (lons[:, None] >= lon_w[None, :]) & (lons[:, None] < lon_e[None, :])
    in_lat = (lats[:, None] >= lat_s[None, :]) & (lats[:, None] < lat_n[None, :])
    lon_hit = in_lon.sum(axis=1)
    lat_hit = in_lat.sum(axis=1)
    valid = (lon_hit == 1) & (lat_hit == 1)

    lon_i = (in_lon * np.arange(n_longitude, dtype=np.int64)).sum(axis=1)
    lat_i = (in_lat * np.arange(n_latitude, dtype=np.int64)).sum(axis=1)
    out = np.full(lons.shape[0], -1, dtype=np.int64)
    out[valid] = lat_i[valid] * n_longitude + lon_i[valid]
    return out
