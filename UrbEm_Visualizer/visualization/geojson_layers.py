from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer

from UrbEm_Visualizer.visualization.load_run import RunContext
from UrbEm_Visualizer.visualization.map_config import load_map_config, sector_viz_meta


def _cell_half_sizes(df: pd.DataFrame) -> tuple[float, float]:
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 100.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 100.0
    return dx * 0.5, dy * 0.5


def _color_stops() -> list[tuple[float, str]]:
    cfg = load_map_config()
    stops = cfg.get("area_colormap", {}).get("stops") or []
    out = []
    for row in stops:
        out.append((float(row[0]), str(row[1])))
    if not out:
        out = [(0.0, "#ffffcc"), (1.0, "#800026")]
    return out


def value_to_color(value: float, vmin: float, vmax: float) -> str:
    stops = _color_stops()
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            if t1 <= t0:
                return c1
            f = (t - t0) / (t1 - t0)
            return _lerp_hex(c0, c1, f)
    return stops[-1][1]


def _lerp_hex(c0: str, c1: str, f: float) -> str:
    def parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r0, g0, b0 = parse(c0)
    r1, g1, b1 = parse(c1)
    r = int(r0 + (r1 - r0) * f)
    g = int(g0 + (g1 - g0) * f)
    b = int(b0 + (b1 - b0) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def area_geojson(
    ctx: RunContext,
    sector_id: str,
    pollutant: str,
    active_sectors: list[str] | None = None,
) -> dict[str, Any]:
    if sector_id == "TOTAL":
        sectors = active_sectors or ctx.sector_ids()
        area_ids = [
            s for s in sectors
            if s != "TOTAL" and ctx.sector_layers(s)["area"]
        ]
        df = ctx.sum_sectors_df(area_ids, "area", pollutant)
    else:
        df = ctx.grid_df(sector_id, "area", pollutant)
    if df.empty:
        return {"type": "FeatureCollection", "features": [], "meta": {"vmin": 0, "vmax": 0}}

    hx, hy = _cell_half_sizes(df)
    to_wgs = ctx._to_wgs
    vals = df["emission"].astype(float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    features = []
    for rec in df.itertuples(index=False):
        cx, cy, v = float(rec.x), float(rec.y), float(rec.emission)
        corners = [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
            (cx - hx, cy - hy),
        ]
        lons, lats = to_wgs.transform([p[0] for p in corners], [p[1] for p in corners])
        coords = [[float(lons[i]), float(lats[i])] for i in range(len(corners))]
        features.append({
            "type": "Feature",
            "properties": {
                "value": v,
                "color": value_to_color(v, vmin, vmax),
                "sector": sector_id,
                "pollutant": pollutant,
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {"vmin": vmin, "vmax": vmax, "pollutant": pollutant, "sector": sector_id},
    }


def _round_key(lon: float, lat: float) -> tuple[float, float]:
    return (round(lon, 5), round(lat, 5))


def _merge_point_shape(current: str, new: str) -> str:
    rank = {"box": 0, "diamond": 1, "sphere": 2}
    return new if rank.get(new, 0) > rank.get(current, 0) else current


def _real_sector_ids(sector_ids: list[str]) -> list[str]:
    return [s for s in sector_ids if s != "TOTAL"]


def points_geojson(
    ctx: RunContext,
    sector_ids: list[str],
    pollutant: str,
) -> dict[str, Any]:
    buckets: dict[tuple[float, float], dict[str, Any]] = {}

    for sid in _real_sector_ids(sector_ids):
        if not ctx.sector_layers(sid)["point"]:
            continue
        meta = sector_viz_meta(sid)
        if ctx.use_facility_points(sid):
            df = ctx.facility_points_df(sid, pollutant)
            for rec in df.itertuples(index=False):
                lon, lat = float(rec.lon), float(rec.lat)
                key = _round_key(lon, lat)
                if key not in buckets:
                    buckets[key] = {
                        "lon": lon,
                        "lat": lat,
                        "sectors": [],
                        "emissions": {},
                        "accents": [],
                        "point_shape": "box",
                    }
                b = buckets[key]
                if sid not in b["sectors"]:
                    b["sectors"].append(sid)
                    b["accents"].append(meta.get("accent", "#4f7cff"))
                b["emissions"][sid] = b["emissions"].get(sid, 0.0) + float(rec.emission)
                b["point_shape"] = _merge_point_shape(
                    b.get("point_shape", "box"),
                    getattr(rec, "point_shape", "box"),
                )
            continue

        df = ctx.grid_df(sid, "point", pollutant)
        if df.empty:
            continue
        to_wgs = ctx._to_wgs
        for rec in df.itertuples(index=False):
            lon, lat = to_wgs.transform(float(rec.x), float(rec.y))
            key = _round_key(lon, lat)
            if key not in buckets:
                buckets[key] = {
                    "lon": lon,
                    "lat": lat,
                    "sectors": [],
                    "emissions": {},
                    "accents": [],
                }
            b = buckets[key]
            if sid not in b["sectors"]:
                b["sectors"].append(sid)
                b["accents"].append(meta.get("accent", "#4f7cff"))
            b["emissions"][sid] = b["emissions"].get(sid, 0.0) + float(rec.emission)
            b["point_shape"] = b.get("point_shape", "box")

    features = []
    for i, b in enumerate(buckets.values()):
        total = sum(b["emissions"].values())
        features.append({
            "type": "Feature",
            "properties": {
                "facility_id": f"pt_{i}",
                "sectors": b["sectors"],
                "accents": b["accents"],
                "emissions": b["emissions"],
                "emission": total,
                "pollutant": pollutant,
                "point_shape": b.get("point_shape", "box"),
            },
            "geometry": {"type": "Point", "coordinates": [b["lon"], b["lat"]]},
        })
    return {"type": "FeatureCollection", "features": features}


def domain_bbox_geojson(ctx: RunContext) -> dict[str, Any]:
    from UrbEm_Visualizer.visualization.load_run import domain_corners_wgs84

    ring = domain_corners_wgs84(ctx.domain)
    return {
        "type": "Feature",
        "properties": {"name": "Domain", "crs": str(ctx.domain.get("crs", ""))},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def domain_wgs84_from_ctx(ctx: RunContext) -> tuple[float, float, float, float]:
    from UrbEm_Visualizer.visualization.load_run import domain_wgs84

    return domain_wgs84(ctx.domain)


def match_lines_geojson(
    ctx: RunContext,
    sector_ids: list[str],
    pollutant: str,
) -> dict[str, Any]:
    from UrbEm_Visualizer.downscaling.point_meta import (
        appointed_meta,
        facility_links,
        load_match_sidecar,
        sidecar_pollutant_mass,
    )

    w, s, e, n = domain_wgs84_from_ctx(ctx)

    def _in_dom(lon: float, lat: float) -> bool:
        return w <= lon <= e and s <= lat <= n

    features = []
    for sid in _real_sector_ids(sector_ids):
        if not ctx.sector_layers(sid)["point"]:
            continue
        link_path = ctx._sector_link_path(sid)
        if not link_path:
            continue
        accent = sector_viz_meta(sid).get("accent", "#4f7cff")
        sidecar = load_match_sidecar(link_path)
        for pid_str, rec in sidecar.items():
            if rec.get("matched") != "yes":
                continue
            clon, clat = float(rec["cams_lon"]), float(rec["cams_lat"])
            for lk in facility_links(rec):
                flon, flat = lk.get("facility_lon"), lk.get("facility_lat")
                if flon is None or flat is None:
                    continue
                flon, flat = float(flon), float(flat)
                if not (_in_dom(clon, clat) or _in_dom(flon, flat)):
                    continue
                attr = lk.get("attributed_pollutants") or {}
                mass = sidecar_pollutant_mass(attr, pollutant) if attr else 0.0
                if mass <= 0 and attr:
                    continue
                meta = appointed_meta(rec, lk)
                features.append({
                    "type": "Feature",
                    "properties": {
                        "sector": sid,
                        "accent": accent,
                        "cams_point_id": int(pid_str),
                        "dataset": meta.get("dataset"),
                        "facility_id": meta.get("facility_id"),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[clon, clat], [flon, flat]],
                    },
                })
    return {"type": "FeatureCollection", "features": features}


def facility_detail(
    ctx: RunContext,
    lon: float,
    lat: float,
    pollutant: str,
    sector_ids: list[str],
) -> dict[str, Any]:
    from UrbEm_Visualizer.visualization.map_config import sector_viz_meta
    from UrbEm_Visualizer.visualization.scale import _fmt_sci

    facilities = []
    for sid in _real_sector_ids(sector_ids):
        if not ctx.sector_layers(sid)["point"]:
            continue
        rec = ctx.appointed_facility_at(sid, lon, lat)
        if rec:
            sm = sector_viz_meta(sid)
            facilities.append({
                **rec,
                "sector": sid,
                "sector_label": sm.get("tree_label", sid),
            })
            continue
        if ctx.use_facility_points(sid):
            df = ctx.facility_points_df(sid, pollutant)
            for row in df.itertuples(index=False):
                if _round_key(float(row.lon), float(row.lat)) == _round_key(lon, lat):
                    fac_in = None
                    if hasattr(row, "facility_lon") and getattr(row, "facility_lon", None) is not None:
                        try:
                            fac_in = ctx.facility_in_domain(
                                float(row.facility_lon), float(row.facility_lat),
                            )
                        except (TypeError, ValueError):
                            pass
                    facilities.append({
                        "sector_id": sid,
                        "sector": sid,
                        "sector_label": sector_viz_meta(sid).get("tree_label", sid),
                        "match_status": str(getattr(row, "match_status", "unknown")),
                        "mass_outcome": ctx.mass_outcome_line(str(getattr(row, "match_status", "unknown"))),
                        "partial_match_notice": ctx.partial_match_notice(
                            str(getattr(row, "match_status", "unknown")),
                            facility_in_domain=fac_in,
                        ),
                        "pollutants": [{
                            "pollutant": pollutant,
                            "emission": float(row.emission),
                            "emission_label": _fmt_sci(float(row.emission)),
                        }],
                    })
                    break

    if not facilities:
        tr = Transformer.from_crs("EPSG:4326", str(ctx.domain["crs"]), always_xy=True)
        x, y = tr.transform(lon, lat)
        rows = []
        for sid in _real_sector_ids(sector_ids):
            if not ctx.sector_layers(sid)["point"]:
                continue
            df = ctx.grid_df(sid, "point", pollutant)
            if df.empty:
                continue
            to_wgs = ctx._to_wgs
            for rec in df.itertuples(index=False):
                plon, plat = to_wgs.transform(float(rec.x), float(rec.y))
                if _round_key(plon, plat) == _round_key(lon, lat):
                    rows.append({
                        "sector": sid,
                        "label": sector_viz_meta(sid).get("tree_label", sid),
                        "emission": float(rec.emission),
                    })
        return {
            "pollutant": pollutant,
            "unit": ctx.unit,
            "lon": lon,
            "lat": lat,
            "facilities": [],
            "sectors": rows,
            "total": sum(r["emission"] for r in rows),
        }

    primary = facilities[0]
    all_pollutants: dict[str, float] = {}
    for fac in facilities:
        for p in fac.get("pollutants") or []:
            pol = p["pollutant"]
            all_pollutants[pol] = all_pollutants.get(pol, 0.0) + float(p["emission"])

    pollutant_rows = [
        {
            "pollutant": pol,
            "emission": val,
            "emission_label": _fmt_sci(val),
        }
        for pol, val in sorted(all_pollutants.items(), key=lambda x: ctx.pollutants.index(x[0]) if x[0] in ctx.pollutants else 999)
    ]

    return {
        "pollutant": pollutant,
        "unit": ctx.unit,
        "lon": lon,
        "lat": lat,
        "match_status": primary.get("match_status"),
        "mass_outcome": primary.get("mass_outcome") or ctx.mass_outcome_line(primary.get("match_status")),
        "partial_match_notice": primary.get("partial_match_notice")
        or ctx.partial_match_notice(primary.get("match_status")),
        "dataset": primary.get("dataset"),
        "facility_name": primary.get("facility_name"),
        "facility_id": primary.get("facility_id"),
        "details": primary.get("details") or [],
        "match_distance_km": primary.get("match_distance_km"),
        "cams_point_id": primary.get("cams_point_id"),
        "cams_lon": primary.get("cams_lon"),
        "cams_lat": primary.get("cams_lat"),
        "pollutants": pollutant_rows,
        "sectors": [
            {
                "sector": f["sector"],
                "label": f["sector_label"],
            }
            for f in facilities
        ],
        "facilities": facilities,
    }
