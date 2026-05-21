from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import Point

from PROXY_V2.core import log

def _cams_mass(cams: dict[str, Any]) -> float:
    pols = cams.get("pollutants") or {}
    if not pols:
        return 1.0
    vals = [float(v) for v in pols.values() if float(v) > 0]
    return float(sum(vals)) if vals else 1.0


def _facility_lon_lat(match_row: dict[str, Any]) -> tuple[float, float] | None:
    """WGS84 (lon, lat) of the linked facility when matched, else nearest diagnostic point."""
    if match_row.get("matched") == "yes":
        for key in (
            "corine_facility_info",
            "uwwtd_facility_info",
            "eprtr_point_info",
            "jrc_point_info",
            "osm_facility_info",
        ):
            info = match_row.get(key)
            if isinstance(info, dict):
                lon, lat = info.get("lon"), info.get("lat")
                if lon is not None and lat is not None:
                    return float(lon), float(lat)
    for key in (
        "uwwtd_facility_info",
        "eprtr_point_info",
        "jrc_point_info",
        "osm_facility_info",
        "corine_facility_info",
    ):
        info = match_row.get(key)
        if isinstance(info, dict):
            lon, lat = info.get("lon"), info.get("lat")
            if lon is not None and lat is not None:
                return float(lon), float(lat)
    return None


def write_cams_facility_link_tif(
    matches: dict[int, dict[str, Any]],
    out_tif: Path,
    *,
    crs: str,
    resolution_m: float,
    pad_m: float,
) -> Path:
    """
    Two-band GeoTIFF on a grid covering all CAMS + linked facility points.

    Band 1 (CAMS): every CAMS point mass at its WGS84 location (all entries in *matches*).
    Band 2 (Facilities): same mass at the linked facility location (JRC, EPRTR/LCP, or OSM aviation
    match point) when ``matched == "yes"``.
    Matching pairs share the same burn value so linked pixels can be traced pairwise.
    """
    if not matches:
        raise ValueError("matches is empty")

    # Parse options, initialize geometry list
    res = float(resolution_m)
    pad = float(pad_m)
    pts: list[Point] = []

    # Collect all CAMS points and all linked facility points 
    for pid, m in matches.items():
        c = m["cams"]
        # Always add CAMS location
        pts.append(Point(float(c["longitude"]), float(c["latitude"])))
        fac = _facility_lon_lat(m)
        if fac is not None:
            # Add facility (JRC/EPRTR) location if present
            pts.append(Point(fac[0], fac[1]))

    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326").to_crs(crs)
    minx, miny, maxx, maxy = (float(x) for x in gdf.total_bounds)

    # Pad the bounding box to ensure room around points
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    # Compute raster dimensions
    width = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Prepare raster bands for CAMS and facilities
    band_cams = np.zeros((height, width), dtype=np.float32)
    band_fac = np.zeros((height, width), dtype=np.float32)
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _accum(lon: float, lat: float, arr: np.ndarray, val: float) -> None:
        x, y = to_crs.transform(lon, lat)
        r, c = rowcol(transform, x, y)
        ri, ci = int(r), int(c)
        if 0 <= ri < height and 0 <= ci < width:
            arr[ri, ci] += float(val)

    n_linked = 0
    # Burn all CAMS points into band 1; burn linked facilities into band 2
    for pid, m in matches.items():
        c = m["cams"]
        v = _cams_mass(c)
        _accum(float(c["longitude"]), float(c["latitude"]), band_cams, v)
        if m.get("matched") != "yes":
            continue
        fac = _facility_lon_lat(m)
        if fac is not None:
            _accum(fac[0], fac[1], band_fac, v)
            n_linked += 1

    # Write the stack (2 bands) out to a GeoTIFF
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack([band_cams, band_fac], axis=0)
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=0.0,
    ) as dst:
        dst.write(stacked)
        dst.set_band_description(1, "CAMS_point_mass")
        dst.set_band_description(2, "linked_facility_mass")
        dst.update_tags(
            description="CAMS and linked facility masses (JRC/EPRTR/OSM/CORINE); equal values mark a pair",
            n_cams_points=str(len(matches)),
            n_matched_yes=str(n_linked),
            crs=crs,
        )

    log.info(f"Wrote 2-band link GeoTIFF: {out_tif} ({len(matches)} CAMS, {n_linked} matched)")
    return out_tif
