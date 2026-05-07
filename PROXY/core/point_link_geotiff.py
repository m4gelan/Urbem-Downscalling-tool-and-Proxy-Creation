"""Two-band GeoTIFF: CAMS point mass on the ref grid vs linked facility mass on the ref grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

from PROXY.core.io import write_geotiff


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
