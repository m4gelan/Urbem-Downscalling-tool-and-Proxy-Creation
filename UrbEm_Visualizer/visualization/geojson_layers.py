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
                    }
                b = buckets[key]
                if sid not in b["sectors"]:
                    b["sectors"].append(sid)
                    b["accents"].append(meta.get("accent", "#4f7cff"))
                b["emissions"][sid] = b["emissions"].get(sid, 0.0) + float(rec.emission)
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
            },
            "geometry": {"type": "Point", "coordinates": [b["lon"], b["lat"]]},
        })
    return {"type": "FeatureCollection", "features": features}


def domain_bbox_geojson(ctx: RunContext) -> dict[str, Any]:
    w, s, e, n = domain_wgs84_from_ctx(ctx)
    coords = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    return {
        "type": "Feature",
        "properties": {"name": "Domain"},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def domain_wgs84_from_ctx(ctx: RunContext) -> tuple[float, float, float, float]:
    from UrbEm_Visualizer.visualization.load_run import domain_wgs84

    return domain_wgs84(ctx.domain)


def facility_detail(
    ctx: RunContext,
    lon: float,
    lat: float,
    pollutant: str,
    sector_ids: list[str],
) -> dict[str, Any]:
    tr = Transformer.from_crs("EPSG:4326", str(ctx.domain["crs"]), always_xy=True)
    x, y = tr.transform(lon, lat)
    rows = []
    for sid in _real_sector_ids(sector_ids):
        if not ctx.sector_layers(sid)["point"]:
            continue
        if ctx.use_facility_points(sid):
            df = ctx.facility_points_df(sid, pollutant)
            for rec in df.itertuples(index=False):
                if _round_key(float(rec.lon), float(rec.lat)) == _round_key(lon, lat):
                    rows.append({
                        "sector": sid,
                        "label": sector_viz_meta(sid).get("tree_label", sid),
                        "emission": float(rec.emission),
                    })
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
        "sectors": rows,
        "total": sum(r["emission"] for r in rows),
    }
