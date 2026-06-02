from __future__ import annotations

from typing import Any

# Each entry: human name, ISO-3166 alpha-3 (CAMS ``country_id``), alpha-2, NUTS CNTR_CODE-style.
_COUNTRY_ROWS: tuple[dict[str, str], ...] = (
    {"full_name": "Austria", "ISO3": "AUT", "Abbreviation": "AT", "other": "AT"},
    {"full_name": "Belgium", "ISO3": "BEL", "Abbreviation": "BE", "other": "BE"},
    {"full_name": "Bulgaria", "ISO3": "BGR", "Abbreviation": "BG", "other": "BG"},
    {"full_name": "Croatia", "ISO3": "HRV", "Abbreviation": "HR", "other": "HR"},
    {"full_name": "Cyprus", "ISO3": "CYP", "Abbreviation": "CY", "other": "CY"},
    {"full_name": "Czechia", "ISO3": "CZE", "Abbreviation": "CZ", "other": "CZ"},
    {"full_name": "Denmark", "ISO3": "DNK", "Abbreviation": "DK", "other": "DK"},
    {"full_name": "Estonia", "ISO3": "EST", "Abbreviation": "EE", "other": "EE"},
    {"full_name": "Finland", "ISO3": "FIN", "Abbreviation": "FI", "other": "FI"},
    {"full_name": "France", "ISO3": "FRA", "Abbreviation": "FR", "other": "FR"},
    {"full_name": "Germany", "ISO3": "DEU", "Abbreviation": "DE", "other": "DE"},
    {"full_name": "Greece", "ISO3": "GRC", "Abbreviation": "GR", "other": "EL"},
    {"full_name": "Hungary", "ISO3": "HUN", "Abbreviation": "HU", "other": "HU"},
    {"full_name": "Ireland", "ISO3": "IRL", "Abbreviation": "IE", "other": "IE"},
    {"full_name": "Italy", "ISO3": "ITA", "Abbreviation": "IT", "other": "IT"},
    {"full_name": "Latvia", "ISO3": "LVA", "Abbreviation": "LV", "other": "LV"},
    {"full_name": "Lithuania", "ISO3": "LTU", "Abbreviation": "LT", "other": "LT"},
    {"full_name": "Luxembourg", "ISO3": "LUX", "Abbreviation": "LU", "other": "LU"},
    {"full_name": "Malta", "ISO3": "MLT", "Abbreviation": "MT", "other": "MT"},
    {"full_name": "Netherlands", "ISO3": "NLD", "Abbreviation": "NL", "other": "NL"},
    {"full_name": "Poland", "ISO3": "POL", "Abbreviation": "PL", "other": "PL"},
    {"full_name": "Portugal", "ISO3": "PRT", "Abbreviation": "PT", "other": "PT"},
    {"full_name": "Romania", "ISO3": "ROU", "Abbreviation": "RO", "other": "RO"},
    {"full_name": "Slovakia", "ISO3": "SVK", "Abbreviation": "SK", "other": "SK"},
    {"full_name": "Slovenia", "ISO3": "SVN", "Abbreviation": "SI", "other": "SI"},
    {"full_name": "Switzerland", "ISO3": "CHE", "Abbreviation": "CH", "other": "CH"},
    {"full_name": "Spain", "ISO3": "ESP", "Abbreviation": "ES", "other": "ES"},
    {"full_name": "Sweden", "ISO3": "SWE", "Abbreviation": "SE", "other": "SE"},
)

# Each entry: CAMS / sector config spelling, NetCDF variable name (``small_cap``), reported-emissions
# workbook ``POLLUTANT`` column (``alpha``) — CEIP-style strings as in the Excel export.
_POLLUTANT_ROWS: tuple[dict[str, str], ...] = (
    {"CAMS": "NH3", "small_cap": "nh3", "alpha": "NH3"},
    {"CAMS": "CO", "small_cap": "co", "alpha": "CO"},
    {"CAMS": "NMVOC", "small_cap": "nmvoc", "alpha": "NMVOC"},
    {"CAMS": "NOx", "small_cap": "nox", "alpha": "NOx"},
    {"CAMS": "PM10", "small_cap": "pm10", "alpha": "PM10"},
    {"CAMS": "PM2.5", "small_cap": "pm2_5", "alpha": "PM2.5"},
    {"CAMS": "SOx", "small_cap": "sox", "alpha": "SOx"},
    {"CAMS": "SO2", "small_cap": "sox", "alpha": "SOx"},
    {"CAMS": "CH4", "small_cap": "ch4", "alpha": "CH4"},
    {"CAMS": "CO2", "small_cap": "co2", "alpha": "CO2"},
)


def _resolve_pollutant_row(label: str) -> dict[str, str]:
    """Match *label* to a pollutant row (CAMS config, workbook alpha, or NetCDF ``small_cap``)."""
    q = str(label).strip()
    if not q:
        raise ValueError("pollutant label is empty")
    collapsed = q.upper().replace(" ", "").replace(".", "")
    if collapsed in ("PM25", "PM2_5"):
        q = "PM2.5"
    qu = q.upper()
    for row in _POLLUTANT_ROWS:
        cam = row["CAMS"]
        if q == cam or qu == cam.upper() or q.casefold() == cam.casefold():
            return row
        if q.casefold() == row["alpha"].casefold():
            return row
        sc = row["small_cap"]
        if q == sc or qu == sc.upper():
            return row
    raise ValueError(f"Unknown pollutant {label!r}; add a row to _POLLUTANT_ROWS in alias.py")


def resolve_country_profile(country: str) -> dict[str, str]:
    """Return ``full_name``, ``ISO3``, ``Abbreviation``, ``other`` for *country*.

    ``other`` is the NUTS / Eurostat CNTR-style two-letter code (e.g. Greece ``EL``), used by
    EEA datasets such as UWWTD ``rptMStateKey``; where it matches ISO alpha-2 it equals ``Abbreviation``.
    """
    q = str(country).strip()
    qu = q.upper()
    ql = q.casefold()
    for row in _COUNTRY_ROWS:
        if qu in {row["ISO3"].upper(), row["Abbreviation"].upper(), row["other"].upper()}:
            return dict(row)
        if ql == row["full_name"].casefold():
            return dict(row)
    raise ValueError(f"Unknown country {country!r}; extend proxy.core.alias._COUNTRY_ROWS")


def resolve_osm_filepath(rel_path: str | Path, country_profile: dict[str, str]) -> str:
    """Replace ``Country`` in sector OSM config paths with ``country_profile['full_name']``."""
    country = str(country_profile["full_name"]).strip()
    if not country:
        raise ValueError("country_profile['full_name'] is empty")
    return str(rel_path).replace("\\", "/").replace("Country", country)


def cams_pollutant_var(label: str) -> str:
    """Map sector config pollutant label to CAMS NetCDF variable name (``small_cap``)."""
    return _resolve_pollutant_row(label)["small_cap"]


def workbook_pollutant_label(label: str) -> str:
    """Exact ``POLLUTANT`` string for the reported-emissions workbook (``alpha`` column)."""
    return _resolve_pollutant_row(label)["alpha"]


def normalize_workbook_pollutant_cell(cell: str) -> str:
    """Map a workbook ``POLLUTANT`` cell to the canonical ``alpha`` string (e.g. ``NOx``)."""
    s = str(cell).strip()
    if not s:
        raise ValueError("workbook POLLUTANT cell is empty")
    for row in _POLLUTANT_ROWS:
        if s.casefold() == row["alpha"].casefold():
            return row["alpha"]
        if s.casefold() == row["CAMS"].casefold():
            return row["alpha"]
        if s.casefold() == row["small_cap"].casefold():
            return row["alpha"]
    raise ValueError(
        f"Unknown workbook POLLUTANT {cell!r}; add a row to _POLLUTANT_ROWS or fix the sheet"
    )


def iso3_from_workbook_country_token(tok: str) -> str:
    """Map workbook ``COUNTRY`` (``Abbreviation`` or ``EU27`` / ``EU11``) to ISO3 or aggregate code."""
    t = str(tok).strip().upper()
    if not t:
        raise ValueError("workbook COUNTRY token is empty")
    if t in ("EU27", "EU11"):
        return t
    for row in _COUNTRY_ROWS:
        if row["Abbreviation"].upper() == t:
            return str(row["ISO3"]).strip().upper()
    raise ValueError(
        f"Unknown workbook COUNTRY {tok!r}; expected member-state Abbreviation (e.g. GR) or EU27/EU11"
    )


_GAMMA_SPLIT_HIGH = frozenset({"BE", "NL"})
_GAMMA_SPLIT_MID = frozenset({"DE", "DK", "ES", "FR", "IT", "IE", "PL", "PT", "UK", "GB"})


def gamma_split_manure(country_profile: dict[str, str]) -> float:
    """Velthof pig-split factor (fodder vs non-fodder manure pools) by member state."""
    ab = str(country_profile["Abbreviation"]).strip().upper()
    ot = str(country_profile["other"]).strip().upper()
    if ab in _GAMMA_SPLIT_HIGH or ot in _GAMMA_SPLIT_HIGH:
        return 0.75
    if ab in _GAMMA_SPLIT_MID or ot in _GAMMA_SPLIT_MID:
        return 0.50
    return 0.25


def cams_country_index_from_iso3(ds: Any, iso3: str) -> int:
    """1-based ``country_index`` for CAMS-REG ``country_id`` ISO3 (e.g. GRC)."""
    want = str(iso3).strip().upper()
    if "country_id" not in ds:
        raise ValueError("CAMS dataset has no country_id")
    raw = ds["country_id"].values
    ids = [str(x.decode("utf-8") if isinstance(x, bytes) else x).strip().upper() for x in raw]
    if want not in ids:
        raise ValueError(f"ISO3 {want!r} not in CAMS country_id list (sample: {ids[:8]})")
    return ids.index(want) + 1
