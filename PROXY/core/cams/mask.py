"""CAMS source-row masks by GNFR, country, and source geometry type.

This module owns reusable CAMS mask logic that operates on source rows in the
CAMS NetCDF dataset. Inputs are an open CAMS dataset, a country ISO3 code, a
GNFR category, optional source-type filters, and an optional WGS84 domain box.
Outputs are boolean masks aligned to CAMS source rows for downstream maps and
sector diagnostics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from PROXY.core.cams.domain import country_index_1based, domain_mask_wgs84
from PROXY.core.cams.gnfr import gnfr_code_to_index


def cams_gnfr_country_source_mask(
    ds: xr.Dataset,
    iso3: str,
    *,
    gnfr: str = "C",
    source_types: tuple[str, ...] | None = None,
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """True where CAMS lists ``gnfr`` for ``iso3`` with the requested source geometry types."""
    cidx = country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    emis_idx = gnfr_code_to_index(gnfr)
    dom = domain_mask_wgs84(lon, lat, ci, cidx, domain_bbox_wgs84)
    mask = dom & (emis == emis_idx)
    types = list(source_types or ("area",))
    if "area" in types and "point" not in types:
        mask = mask & (st == 1)
    elif "point" in types and "area" not in types:
        mask = mask & (st == 2)
    elif "area" in types and "point" in types:
        mask = mask & ((st == 1) | (st == 2))
    return mask


def other_combustion_area_mask(
    ds: xr.Dataset,
    iso3: str,
    *,
    gnfr: str = "C",
    source_types: tuple[str, ...] | None = None,
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Alias for :func:`cams_gnfr_country_source_mask` (GNFR C other combustion defaults)."""
    return cams_gnfr_country_source_mask(
        ds,
        iso3,
        gnfr=gnfr,
        source_types=source_types,
        domain_bbox_wgs84=domain_bbox_wgs84,
    )


__all__ = ["cams_gnfr_country_source_mask", "other_combustion_area_mask"]
