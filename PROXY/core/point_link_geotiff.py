"""Two-band GeoTIFF: CAMS point mass on the ref grid vs linked facility mass on the ref grid."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import Point

from PROXY.core.io import write_geotiff

# Burning onto full EU CORINE / population grids allocates height x width float32 arrays and can
# exhaust RAM or run for a very long time. Prefer :func:`write_cams_facility_link_geotiff_match_extent`.
_MAX_REF_GRID_PIXELS_FOR_POINT_BURN = 50_000_000


def write_cams_facility_link_geotiff(
    *,
    matches_df: pd.DataFrame,
    ref_weights_tif: Path,
    out_tif: Path,
    value_column: str = "cams_pollutant_value",
) -> Path:
    """
    Burn each matched pair onto the **same CRS/transform** as ``ref_weights_tif``.

    Band 1 accumulates ``value_column`` at the pixel under each CAMS WGS84 point.
    Band 2 accumulates the same value at the pixel under each linked facility WGS84 point.

    Multiple matches in one pixel sum. Uses EPSG:4326 source coordinates and the
    reference raster CRS for row/column lookup.
    """
    required = {
        "cams_longitude",
        "cams_latitude",
        "facility_longitude",
        "facility_latitude",
        value_column,
    }
    miss = required - set(matches_df.columns)
    if miss:
        raise ValueError(f"matches dataframe missing columns: {sorted(miss)}")

    with rasterio.open(ref_weights_tif) as src:
        if src.crs is None:
            raise ValueError(f"Reference raster has no CRS: {ref_weights_tif}")
        crs_s = src.crs.to_string()
        transform = src.transform
        h, w = int(src.height), int(src.width)

    npixels = h * w
    if npixels > _MAX_REF_GRID_PIXELS_FOR_POINT_BURN:
        raise MemoryError(
            f"Reference raster {ref_weights_tif.name} is {w}x{h} pixels (~{npixels / 1e6:.1f} Mpx); "
            "use write_cams_facility_link_geotiff_match_extent or a local/cropped reference TIFF."
        )

    band_cams = np.zeros((h, w), dtype=np.float32)
    band_fac = np.zeros((h, w), dtype=np.float32)

    to_ref = Transformer.from_crs("EPSG:4326", crs_s, always_xy=True)

    def _accum(lon: float, lat: float, arr: np.ndarray, val: float) -> None:
        x, y = to_ref.transform(float(lon), float(lat))
        try:
            r, c = rowcol(transform, x, y)
        except Exception:
            return
        if 0 <= int(r) < h and 0 <= int(c) < w:
            arr[int(r), int(c)] += float(val)

    for row in matches_df.itertuples(index=False):
        v = float(getattr(row, value_column, 0.0) or 0.0)
        if not np.isfinite(v) or v <= 0.0:
            continue
        _accum(
            float(row.cams_longitude),
            float(row.cams_latitude),
            band_cams,
            v,
        )
        _accum(
            float(row.facility_longitude),
            float(row.facility_latitude),
            band_fac,
            v,
        )

    stacked = np.stack([band_cams, band_fac], axis=0)
    write_geotiff(
        path=out_tif,
        array=stacked,
        crs=crs_s,
        transform=transform,
        nodata=0.0,
        band_descriptions=[
            "cams_point_mass_sum_at_ref_pixel",
            "linked_facility_mass_sum_at_ref_pixel",
        ],
        tags={
            "description": "CAMS-to-facility link masses on reference grid",
            "value_column": value_column,
        },
    )
    return out_tif


def write_cams_facility_link_geotiff_match_extent(
    *,
    matches_df: pd.DataFrame,
    out_tif: Path,
    resolution_m: float = 1000.0,
    pad_m: float = 15000.0,
    crs: str = "EPSG:3035",
    value_column: str = "cams_pollutant_value",
) -> Path:
    """Burn matches onto a **small** grid covering CAMS + facility points (plus padding).

    Avoids allocating arrays the size of continental CORINE / population rasters.
    """
    required = {
        "cams_longitude",
        "cams_latitude",
        "facility_longitude",
        "facility_latitude",
        value_column,
    }
    miss = required - set(matches_df.columns)
    if miss:
        raise ValueError(f"matches dataframe missing columns: {sorted(miss)}")
    res = float(resolution_m)
    if not np.isfinite(res) or res <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m!r}")
    pad = float(pad_m)
    if not np.isfinite(pad) or pad < 0:
        pad = 0.0

    pts: list[Point] = []
    for row in matches_df.itertuples(index=False):
        pts.append(Point(float(row.cams_longitude), float(row.cams_latitude)))
        pts.append(Point(float(row.facility_longitude), float(row.facility_latitude)))
    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    g2 = gdf.to_crs(crs)
    minx, miny, maxx, maxy = (float(x) for x in g2.total_bounds)
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    width = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    band_cams = np.zeros((height, width), dtype=np.float32)
    band_fac = np.zeros((height, width), dtype=np.float32)

    to_ref = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _accum(lon: float, lat: float, arr: np.ndarray, val: float) -> None:
        x, y = to_ref.transform(float(lon), float(lat))
        try:
            r, c = rowcol(transform, x, y)
        except Exception:
            return
        if 0 <= int(r) < height and 0 <= int(c) < width:
            arr[int(r), int(c)] += float(val)

    for row in matches_df.itertuples(index=False):
        v = float(getattr(row, value_column, 0.0) or 0.0)
        if not np.isfinite(v) or v <= 0.0:
            continue
        _accum(float(row.cams_longitude), float(row.cams_latitude), band_cams, v)
        _accum(float(row.facility_longitude), float(row.facility_latitude), band_fac, v)

    stacked = np.stack([band_cams, band_fac], axis=0)
    write_geotiff(
        path=out_tif,
        array=stacked,
        crs=crs,
        transform=transform,
        nodata=0.0,
        band_descriptions=[
            "cams_point_mass_sum_at_ref_pixel",
            "linked_facility_mass_sum_at_ref_pixel",
        ],
        tags={
            "description": "CAMS-to-facility link masses on match-extent grid",
            "value_column": value_column,
            "grid_mode": "match_extent",
        },
    )
    return out_tif
