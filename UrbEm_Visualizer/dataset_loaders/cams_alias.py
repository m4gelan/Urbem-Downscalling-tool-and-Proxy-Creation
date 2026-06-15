from __future__ import annotations

from typing import Any

from UrbEm_Visualizer.pollutants import cams_netcdf_var as cams_pollutant_var


def country_iso3(country: str) -> str:
    from proxy.core.alias import resolve_country_profile

    return str(resolve_country_profile(country)["ISO3"])


def cams_country_index_from_iso3(ds: Any, iso3: str) -> int:
    from proxy.core.alias import cams_country_index_from_iso3 as _idx

    return _idx(ds, iso3)
