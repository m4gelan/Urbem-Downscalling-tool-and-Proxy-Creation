"""Country-index and WGS84 domain helpers for CAMS NetCDF datasets.

Consolidates the identical copies of ``country_index_1based`` in
``PROXY/core/area_allocation.py``, ``PROXY/sectors/J_Waste/cams_waste_grid.py``, and
``PROXY/sectors/C_OtherCombustion/pipeline.py`` (CAMS allocation), plus the ``domain_mask_wgs84``
helper from ``area_allocation.py``. Signatures match the existing callers to allow a
drop-in migration later.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    """1-based position of ``iso3`` in the CAMS dataset's ``country_id`` array.

    Raises :class:`ValueError` if the country is not present.
    """
    if "country_id" not in ds:
        raise ValueError("Missing country_id in CAMS dataset")
    ids = [
        str(x.decode("utf-8") if isinstance(x, bytes) else x).strip().upper()
        for x in ds["country_id"].values
    ]
    key = iso3.strip().upper()
    if key not in ids:
        raise ValueError(f"Country ISO3 {iso3!r} not found in CAMS country_id")
    return ids.index(key) + 1


def domain_mask_wgs84(
    lon: np.ndarray,
    lat: np.ndarray,
    country_idx: np.ndarray,
    country_1based: int,
    bbox_wgs84: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Country filter combined with an optional WGS84 bbox ``(west, south, east, north)``."""
    m = country_idx == int(country_1based)
    if bbox_wgs84 is not None:
        lon0, lat0, lon1, lat1 = bbox_wgs84
        m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    return m
