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


def _link_mass(link: dict[str, Any]) -> float:
    attributed = link.get("attributed_pollutants") or {}
    if attributed:
        vals = [float(v) for v in attributed.values() if float(v) > 0]
        return float(sum(vals)) if vals else 0.0
    info = link.get("facility_info") or {}
    return _cams_mass({"pollutants": info.get("pollutants")})


def _facility_links(match_row: dict[str, Any]) -> list[dict[str, Any]]:
    links = match_row.get("facility_links")
    if isinstance(links, list) and links:
        return links
    if match_row.get("matched") != "yes":
        return []
    for id_key, info_key in (
        ("riurbans_point_id", "riurbans_point_info"),
        ("jrc_point_id", "jrc_point_info"),
        ("eprtr_point_id", "eprtr_point_info"),
        ("osm_facility_id", "osm_facility_info"),
        ("corine_facility_id", "corine_facility_info"),
        ("uwwtd_facility_id", "uwwtd_facility_info"),
    ):
        fid = match_row.get(id_key)
        info = match_row.get(info_key)
        if fid and isinstance(info, dict):
            cams = match_row.get("cams") or {}
            return [{
                "facility_id": fid,
                "facility_info": info,
                "attributed_pollutants": dict(cams.get("pollutants") or {}),
                "scoring_value": match_row.get("scoring_value"),
            }]
    return []


def _facility_lon_lat_from_link(link: dict[str, Any]) -> tuple[float, float] | None:
    info = link.get("facility_info") or {}
    lon, lat = info.get("lon"), info.get("lat")
    if lon is not None and lat is not None:
        return float(lon), float(lat)
    return None


def _facility_lon_lat(match_row: dict[str, Any]) -> tuple[float, float] | None:
    links = _facility_links(match_row)
    if links:
        return _facility_lon_lat_from_link(links[0])
    return None


def extract_facility_meta(match_row: dict[str, Any]) -> dict[str, Any]:
    """Compact facility metadata for sidecar export and UI (first link only)."""
    if match_row.get("matched") != "yes":
        return {}
    links = _facility_links(match_row)
    if not links:
        return {}
    link = links[0]
    info = dict(link.get("facility_info") or {})
    src = str(match_row.get("match_source") or info.get("fallback_source") or "").strip().lower()
    fid = link.get("facility_id")
    label_map = {
        "riurbans": "RI-URBANS",
        "jrc": "JRC",
        "eprtr": "E-PRTR",
        "fallback": info.get("fallback_source", "fallback"),
        "osm": "Airport polygons (OSM)",
        "corine": "Airport polygons (CORINE)",
        "uwwtd": "UWWTD",
    }
    label = label_map.get(src, str(src or "facility"))
    name = (
        info.get("facility_name")
        or info.get("name_g")
        or info.get("name")
        or info.get("installationPartName")
        or ""
    )
    details: list[dict[str, str]] = []
    if fid:
        details.append({"label": "Facility id", "value": str(fid)})
    if src == "jrc":
        if info.get("name_g"):
            details.append({"label": "Unit", "value": str(info["name_g"])})
        if info.get("type_g"):
            details.append({"label": "Fuel / type", "value": str(info["type_g"])})
    elif src in ("eprtr", "fallback"):
        if info.get("facility_name"):
            details.append({"label": "Facility", "value": str(info["facility_name"])})
    elif src in ("osm", "corine"):
        if info.get("name"):
            details.append({"label": "Name", "value": str(info["name"])})
    elif src == "uwwtd":
        if info.get("name"):
            details.append({"label": "Plant", "value": str(info["name"])})

    fac_ll = _facility_lon_lat_from_link(link)
    out: dict[str, Any] = {
        "dataset_key": src or None,
        "dataset": label,
        "facility_id": str(fid) if fid else None,
        "facility_name": str(name).strip() or None,
        "details": details,
        "match_distance_km": link.get("scoring_value"),
        "n_facility_links": len(links),
    }
    if fac_ll:
        out["facility_lon"] = fac_ll[0]
        out["facility_lat"] = fac_ll[1]
    return out


def _sidecar_row(match_row: dict[str, Any]) -> dict[str, Any]:
    cams = match_row["cams"]
    links = _facility_links(match_row)
    out: dict[str, Any] = {
        "cams_lon": float(cams["longitude"]),
        "cams_lat": float(cams["latitude"]),
        "pollutants": {k: float(v) for k, v in (cams.get("pollutants") or {}).items()},
        "matched": str(match_row.get("matched") or "no"),
        "match_source": match_row.get("match_source"),
        "flags": list(match_row.get("flags") or []),
        "facility_links": [
            {
                "facility_id": lk.get("facility_id"),
                "facility_lon": (_facility_lon_lat_from_link(lk) or (None, None))[0],
                "facility_lat": (_facility_lon_lat_from_link(lk) or (None, None))[1],
                "attributed_pollutants": {
                    k: float(v) for k, v in (lk.get("attributed_pollutants") or {}).items()
                },
                "match_distance_km": lk.get("scoring_value"),
            }
            for lk in links
        ],
    }
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

    Band 1 (CAMS): every CAMS point mass at its WGS84 location.
    Band 2 (Facilities): attributed mass at each linked facility (proportional split when multiple).
    """
    if not matches:
        raise ValueError("matches is empty")

    res = float(resolution_m)
    pad = float(pad_m)
    pts: list[Point] = []

    for m in matches.values():
        c = m["cams"]
        pts.append(Point(float(c["longitude"]), float(c["latitude"])))
        for lk in _facility_links(m):
            fac = _facility_lon_lat_from_link(lk)
            if fac is not None:
                pts.append(Point(fac[0], fac[1]))

    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326").to_crs(crs)
    minx, miny, maxx, maxy = (float(x) for x in gdf.total_bounds)
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    width = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

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
    for m in matches.values():
        c = m["cams"]
        _accum(float(c["longitude"]), float(c["latitude"]), band_cams, _cams_mass(c))
        if m.get("matched") != "yes":
            continue
        for lk in _facility_links(m):
            fac = _facility_lon_lat_from_link(lk)
            if fac is None:
                continue
            mass = _link_mass(lk)
            if mass > 0.0:
                _accum(fac[0], fac[1], band_fac, mass)
                n_linked += 1

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
            description="CAMS and linked facility attributed masses",
            n_cams_points=str(len(matches)),
            n_facility_links=str(n_linked),
            crs=crs,
        )

    log.info(f"Wrote 2-band link GeoTIFF: {out_tif} ({len(matches)} CAMS, {n_linked} facility links)")
    write_match_sidecar(matches, out_tif)
    return out_tif
