from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds, rowcol
from shapely.geometry import Point

from proxy.core import log


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


def extract_facility_meta(match_row: dict[str, Any]) -> dict[str, Any]:
    """Compact facility metadata for sidecar export and UI."""
    if match_row.get("matched") != "yes":
        return {}

    src = str(match_row.get("match_source") or "").strip().lower()
    info: dict[str, Any] | None = None
    fid: str | None = None

    if src == "corine" or match_row.get("corine_facility_info"):
        src = src or "corine"
        info = match_row.get("corine_facility_info")
        fid = match_row.get("corine_facility_id")
        label = "Airport polygons (CORINE)"
    elif src == "osm" or match_row.get("osm_facility_info"):
        src = src or "osm"
        info = match_row.get("osm_facility_info")
        fid = match_row.get("osm_facility_id")
        label = "Airport polygons (OSM)"
    elif src == "uwwtd" or match_row.get("uwwtd_facility_info"):
        src = src or "uwwtd"
        info = match_row.get("uwwtd_facility_info")
        fid = match_row.get("uwwtd_facility_id")
        label = "UWWTD"
    elif match_row.get("jrc_point_info"):
        src = src or "jrc"
        info = match_row.get("jrc_point_info")
        fid = match_row.get("jrc_point_id")
        label = "JRC"
    elif match_row.get("eprtr_point_info"):
        src = src or "eprtr"
        info = match_row.get("eprtr_point_info")
        fid = match_row.get("eprtr_point_id")
        label = "E-PRTR"
    else:
        return {}

    info = dict(info) if isinstance(info, dict) else {}
    name = (
        info.get("facility_name")
        or info.get("name_g")
        or info.get("name")
        or info.get("installationPartName")
        or ""
    )
    details: list[dict[str, str]] = []
    if src == "jrc":
        if info.get("name_g"):
            details.append({"label": "Unit", "value": str(info["name_g"])})
        if info.get("type_g"):
            details.append({"label": "Fuel / type", "value": str(info["type_g"])})
        if info.get("capacity_g"):
            details.append({"label": "Capacity (MW)", "value": str(info["capacity_g"])})
    elif src == "eprtr":
        if info.get("facility_name"):
            details.append({"label": "Facility", "value": str(info["facility_name"])})
        if info.get("eprtr_annex"):
            details.append({"label": "Annex I activity", "value": str(info["eprtr_annex"])})
        if info.get("reporting_year"):
            details.append({"label": "Reporting year", "value": str(info["reporting_year"])})
    elif src in ("osm", "corine"):
        if info.get("name"):
            details.append({"label": "Name", "value": str(info["name"])})
        if info.get("icao"):
            details.append({"label": "ICAO", "value": str(info["icao"])})
        if info.get("iata"):
            details.append({"label": "IATA", "value": str(info["iata"])})
        if info.get("l3_label"):
            details.append({"label": "CORINE class", "value": str(info["l3_label"])})
    elif src == "uwwtd":
        if info.get("name"):
            details.append({"label": "Plant", "value": str(info["name"])})
        if info.get("uwwtp_id"):
            details.append({"label": "UWWTD id", "value": str(info["uwwtp_id"])})

    if fid:
        details.insert(0, {"label": "Facility id", "value": str(fid)})

    fac_ll = _facility_lon_lat(match_row)
    out = {
        "dataset_key": src,
        "dataset": label,
        "facility_id": str(fid) if fid else None,
        "facility_name": str(name).strip() or None,
        "details": details,
        "match_distance_km": match_row.get("scoring_value"),
    }
    if fac_ll:
        out["facility_lon"] = fac_ll[0]
        out["facility_lat"] = fac_ll[1]
    return out


def _sidecar_row(match_row: dict[str, Any]) -> dict[str, Any]:
    cams = match_row["cams"]
    out: dict[str, Any] = {
        "cams_lon": float(cams["longitude"]),
        "cams_lat": float(cams["latitude"]),
        "pollutants": {k: float(v) for k, v in (cams.get("pollutants") or {}).items()},
        "matched": str(match_row.get("matched") or "no"),
    }
    fac = _facility_lon_lat(match_row)
    if fac:
        out["facility_lon"], out["facility_lat"] = fac
    meta = extract_facility_meta(match_row)
    if meta:
        out.update(meta)
    return out


def write_match_sidecar(matches: dict[int, dict[str, Any]], out_tif: Path) -> Path | None:
    if not matches:
        return None
    rows = {str(pid): _sidecar_row(m) for pid, m in matches.items()}
    out_json = Path(out_tif).with_name(f"{Path(out_tif).stem}_matches.json")
    import json

    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    n_yes = sum(1 for m in matches.values() if m.get("matched") == "yes")
    log.info(f"Wrote point link sidecar: {out_json} ({len(rows)} CAMS, {n_yes} matched)")
    return out_json


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
    write_match_sidecar(matches, out_tif)
    return out_tif
