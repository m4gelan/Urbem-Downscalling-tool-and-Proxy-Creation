"""
CAMS GNFR A (Public Power) area-source mask.

Selects **area** emissions (``source_type_index == 1``) for this sector. Point
sources use a different ``source_type_index`` and are handled by the matching
pipeline, not by :func:`public_power_area_mask`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from PROXY.core.cams.domain import country_index_1based, domain_mask_wgs84

# ``emission_category_index`` value for GNFR A in the CAMS grid file used by PROXY.
GNFR_A_PUBLIC_POWER = 1


def public_power_area_mask(ds: xr.Dataset, iso3: str) -> np.ndarray:
    """
    Boolean mask over the CAMS 1D source dimension: public power **area** sources.

    Conditions (all must hold):

    * ``emission_category_index == GNFR A`` (see :data:`GNFR_A_PUBLIC_POWER`).
    * Source lies in the national **domain** used for area allocation (see
      :func:`PROXY.core.cams.domain.domain_mask_wgs84`).
    * ``country_index`` matches ``iso3`` (1-based index from
      :func:`PROXY.core.cams.domain.country_index_1based`).
    * ``source_type_index == 1`` — **area** source; point sources are excluded.

    Parameters
    ----------
    ds
        CAMS NetCDF with 1D source variables (``longitude_source``, etc.).
    iso3
        ISO-3166-1 alpha-3 string (e.g. ``GRC``), consistent with
        :func:`PROXY.core.raster.country_clip.resolve_cams_country_iso3`.

    Returns
    -------
    1D ``bool`` array, same length as ``ds``'s source axis.
    """
    cidx = country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == GNFR_A_PUBLIC_POWER) & domain_mask_wgs84(
        lon, lat, ci, cidx, None
    )
    return base & (st == 1)
