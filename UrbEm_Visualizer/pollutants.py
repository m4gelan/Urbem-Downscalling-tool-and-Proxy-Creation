from __future__ import annotations

AVAILABLE_POLLUTANTS = [
    "NOx",
    "NMVOC",
    "CO",
    "SO2",
    "NH3",
    "PM10",
    "PM2.5",
]

# CAMS / UI label, NetCDF variable (small_cap), GeoTIFF band description (tif_band).
_POLLUTANT_ROWS: tuple[dict[str, str], ...] = (
    {"CAMS": "NH3", "small_cap": "nh3", "tif_band": "NH3"},
    {"CAMS": "CO", "small_cap": "co", "tif_band": "CO"},
    {"CAMS": "NMVOC", "small_cap": "nmvoc", "tif_band": "NMVOC"},
    {"CAMS": "NOx", "small_cap": "nox", "tif_band": "NOx"},
    {"CAMS": "PM10", "small_cap": "pm10", "tif_band": "PM10"},
    {"CAMS": "PM2.5", "small_cap": "pm2_5", "tif_band": "PM2.5"},
    {"CAMS": "SOx", "small_cap": "sox", "tif_band": "SOx"},
    {"CAMS": "SO2", "small_cap": "sox", "tif_band": "SOx"},
    {"CAMS": "CH4", "small_cap": "ch4", "tif_band": "CH4"},
    {"CAMS": "CO2", "small_cap": "co2", "tif_band": "CO2"},
)


def pollutant_row(label: str) -> dict[str, str]:
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
        if q.casefold() == row["tif_band"].casefold():
            return row
        sc = row["small_cap"]
        if q == sc or qu == sc.upper():
            return row
    raise ValueError(f"Unknown pollutant {label!r}; extend _POLLUTANT_ROWS in pollutants.py")


def cams_netcdf_var(label: str) -> str:
    return pollutant_row(label)["small_cap"]


def pollutant_key(label: str) -> str:
    """Canonical key for matching run-config pollutants to GeoTIFF band names."""
    return pollutant_row(label)["small_cap"]


def band_index_for_pollutant(band_names: list[str], pollutant: str) -> int:
    want = pollutant_key(pollutant)
    for i, name in enumerate(band_names):
        try:
            if pollutant_key(name) == want:
                return i
        except ValueError:
            continue
    raise ValueError(
        f"pollutant {pollutant!r} (key {want!r}) not in area_weights bands: {band_names}"
    )
