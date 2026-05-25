from __future__ import annotations

from typing import Any

from UrbEm_Visualizer.pollutants import cams_netcdf_var as cams_pollutant_var

_COUNTRY_ROWS: tuple[dict[str, str], ...] = (
    {"full_name": "Austria", "ISO3": "AUT"},
    {"full_name": "Belgium", "ISO3": "BEL"},
    {"full_name": "Bulgaria", "ISO3": "BGR"},
    {"full_name": "Croatia", "ISO3": "HRV"},
    {"full_name": "Cyprus", "ISO3": "CYP"},
    {"full_name": "Czechia", "ISO3": "CZE"},
    {"full_name": "Denmark", "ISO3": "DNK"},
    {"full_name": "Estonia", "ISO3": "EST"},
    {"full_name": "Finland", "ISO3": "FIN"},
    {"full_name": "France", "ISO3": "FRA"},
    {"full_name": "Germany", "ISO3": "DEU"},
    {"full_name": "Greece", "ISO3": "GRC"},
    {"full_name": "Hungary", "ISO3": "HUN"},
    {"full_name": "Ireland", "ISO3": "IRL"},
    {"full_name": "Italy", "ISO3": "ITA"},
    {"full_name": "Latvia", "ISO3": "LVA"},
    {"full_name": "Lithuania", "ISO3": "LTU"},
    {"full_name": "Luxembourg", "ISO3": "LUX"},
    {"full_name": "Malta", "ISO3": "MLT"},
    {"full_name": "Netherlands", "ISO3": "NLD"},
    {"full_name": "Poland", "ISO3": "POL"},
    {"full_name": "Portugal", "ISO3": "PRT"},
    {"full_name": "Romania", "ISO3": "ROU"},
    {"full_name": "Slovakia", "ISO3": "SVK"},
    {"full_name": "Slovenia", "ISO3": "SVN"},
    {"full_name": "Spain", "ISO3": "ESP"},
    {"full_name": "Sweden", "ISO3": "SWE"},
)


def country_iso3(country: str) -> str:
    q = str(country).strip()
    qu = q.upper()
    ql = q.casefold()
    for row in _COUNTRY_ROWS:
        if qu == row["ISO3"] or ql == row["full_name"].casefold():
            return row["ISO3"]
    raise ValueError(f"Unknown country {country!r}")


def cams_country_index_from_iso3(ds: Any, iso3: str) -> int:
    want = str(iso3).strip().upper()
    raw = ds["country_id"].values
    ids = [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip().upper() for x in raw]
    if want not in ids:
        raise ValueError(f"ISO3 {want!r} not in CAMS country_id")
    return ids.index(want) + 1
