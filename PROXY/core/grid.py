from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.windows import Window, from_bounds


@dataclass(frozen=True)
class GridSpec:
    crs: str
    resolution_x: float
    resolution_y: float


# NUTS ``CNTR_CODE`` is ISO-3166-1 alpha-2 (plus Eurostat EL for Greece, UK for GB).
# Map common ISO-3166-1 alpha-3 ``--country`` values to that column.
_ISO3_TO_NUTS_CNTR: dict[str, str] = {
    "GRC": "EL",
    "GBR": "UK",
    "DEU": "DE",
    "FRA": "FR",
    "ITA": "IT",
    "ESP": "ES",
    "POL": "PL",
    "ROU": "RO",
    "NLD": "NL",
    "BEL": "BE",
    "CZE": "CZ",
    "PRT": "PT",
    "SWE": "SE",
    "HUN": "HU",
    "AUT": "AT",
    "BGR": "BG",
    "DNK": "DK",
    "FIN": "FI",
    "SVK": "SK",
    "IRL": "IE",
    "HRV": "HR",
    "LTU": "LT",
    "SVN": "SI",
    "LVA": "LV",
    "EST": "EE",
    "CYP": "CY",
    "LUX": "LU",
    "MLT": "MT",
    "NOR": "NO",
    "CHE": "CH",
    "ISL": "IS",
    "LIE": "LI",
    "SRB": "RS",
    "MNE": "ME",
    "MKD": "MK",
    "ALB": "AL",
    "TUR": "TR",
    "UKR": "UA",
    "MDA": "MD",
    "BLR": "BY",
    "RUS": "RU",
}


def resolve_nuts_cntr_code(nuts_country: str) -> str:
    """
    Return the value to use against NUTS ``CNTR_CODE`` (2-letter, upper case).

    Accepts either that code or a common ISO-3 form (e.g. ``GRC`` -> ``EL``).
    """
    c = str(nuts_country).strip().upper()
    if len(c) == 2 and c.isalpha():
        return c
    if len(c) == 3 and c.isalpha():
        return _ISO3_TO_NUTS_CNTR.get(c, c)
    return c


def reference_window_profile(
    *,
    corine_path: Path,
    nuts_gpkg: Path,
    nuts_country: str,
    pad_m: float = 5000.0,
) -> dict[str, Any]:
    """Build a CORINE-aligned reference window around NUTS-2 country union."""
    from rasterio.warp import transform_bounds

    nuts = gpd.read_file(nuts_gpkg)
    if nuts.crs is None:
        raise ValueError("NUTS GeoPackage has no CRS")
    if "LEVL_CODE" not in nuts.columns or "CNTR_CODE" not in nuts.columns:
        raise ValueError("NUTS GeoPackage missing LEVL_CODE and/or CNTR_CODE")

    cntr = resolve_nuts_cntr_code(nuts_country)
    nuts2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    g_cc = nuts2["CNTR_CODE"].astype(str).str.upper().str.strip()
    nuts2 = nuts2[g_cc == cntr].copy()
    if nuts2.empty:
        raise ValueError(
            f"No NUTS2 rows for {nuts_country!r} (NUTS CNTR_CODE used: {cntr!r})"
        )

    union = nuts2.dissolve().geometry.iloc[0]
    if union is None or union.is_empty:
        raise ValueError("NUTS2 union geometry is empty")

    with rasterio.open(corine_path) as src:
        if src.crs is None:
            raise ValueError(f"CORINE has no CRS: {corine_path}")
        crs_string = src.crs.to_string()
        union_proj = gpd.GeoDataFrame(geometry=[union], crs=nuts2.crs).to_crs(src.crs)
        minx, miny, maxx, maxy = union_proj.geometry.iloc[0].bounds
        win = from_bounds(
            minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, transform=src.transform
        )
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            raise ValueError("Reference window is empty after clipping")
        transform = src.window_transform(win)
        height = int(win.height)
        width = int(win.width)
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
        w, s, e, n = transform_bounds(
            src.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
        )

    return {
        "corine_path": corine_path,
        "transform": transform,
        "height": height,
        "width": width,
        "crs": crs_string,
        "window_bounds_3035": (left, bottom, right, top),
        "domain_bbox_wgs84": (w, s, e, n),
    }


def nuts2_for_country(gpkg: Path, nuts_country: str) -> gpd.GeoDataFrame:
    """
    NUTS level-2 features for a single NUTS ``CNTR_CODE`` (e.g. EL) or ISO-3 (e.g. GRC -> EL).
    """
    nuts = gpd.read_file(gpkg)
    if nuts.crs is None:
        raise ValueError("NUTS GeoPackage has no CRS")
    n2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    if "CNTR_CODE" not in n2.columns:
        raise ValueError("GeoPackage missing CNTR_CODE")
    cntr = resolve_nuts_cntr_code(nuts_country)
    cc = n2["CNTR_CODE"].astype(str).str.strip().str.upper()
    n2 = n2[cc == cntr].copy()
    if n2.empty:
        raise ValueError(f"No NUTS-2 rows for {nuts_country!r} (CNTR_CODE used: {cntr!r})")
    return n2

