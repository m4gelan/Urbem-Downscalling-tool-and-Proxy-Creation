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
