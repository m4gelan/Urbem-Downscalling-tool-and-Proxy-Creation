from __future__ import annotations


def country_iso3(country_name: str) -> str:
    from UrbEm_Visualizer.dataset_loaders.cams_alias import country_iso3 as _iso3

    return _iso3(country_name)


def european_countries() -> list[str]:
    from proxy.core.alias import _COUNTRY_ROWS

    return [row["full_name"] for row in _COUNTRY_ROWS]
