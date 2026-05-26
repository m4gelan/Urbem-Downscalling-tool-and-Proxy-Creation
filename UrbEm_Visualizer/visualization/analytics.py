from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Transformer

from UrbEm_Visualizer.visualization.emission_style import default_threshold, threshold_for
from UrbEm_Visualizer.visualization.load_run import RunContext
from UrbEm_Visualizer.visualization.map_config import sector_viz_meta
from UrbEm_Visualizer.visualization.scale import _fmt_sci


def _gini(values: np.ndarray) -> float:
    v = np.sort(values[values > 0].astype(np.float32))
    if v.size < 2:
        return 0.0
    n = v.size
    idx = np.arange(1, n + 1, dtype=np.float32)
    return float((2.0 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1.0) / n)


def _lorenz_curve(values: np.ndarray) -> dict[str, Any]:
    xs = list(range(0, 101, 10))
    v = values[values > 0].astype(np.float32)
    if v.size == 0:
        return {"x_pct": xs, "y_cum_pct": [0.0] * len(xs), "pct90_x": None}
    v = np.sort(v)[::-1]
    total = float(v.sum())
    cum = np.cumsum(v) / total * 100.0
    n = v.size
    ys: list[float] = []
    for xp in xs:
        if xp == 0:
            ys.append(0.0)
            continue
        k = max(1, min(n, int(round(xp / 100.0 * n))))
        ys.append(float(cum[k - 1]))
    pct90_x = None
    for i, y in enumerate(cum):
        if y >= 90.0:
            pct90_x = round(100.0 * (i + 1) / n, 1)
            break
    return {"x_pct": xs, "y_cum_pct": ys, "pct90_x": pct90_x}


def _log_histogram(values: np.ndarray, bins: int = 30) -> dict[str, Any]:
    v = values[values > 0].astype(np.float32)
    if v.size == 0:
        return {"bin_centers": [], "counts": []}
    logv = np.log10(v)
    counts, edges = np.histogram(logv, bins=bins)
    centers = [float(10 ** ((edges[i] + edges[i + 1]) / 2)) for i in range(len(counts))]
    return {"bin_centers": centers, "counts": [int(c) for c in counts]}


def _sector_area_ids(ctx: RunContext) -> list[str]:
    return [sid for sid in ctx.sector_ids() if ctx.sector_layers(sid)["area"]]


def compute_analytics(ctx: RunContext) -> dict[str, Any]:
    sectors = _sector_area_ids(ctx)
    pollutants = list(ctx.pollutants)
    composition: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []
    radar: list[dict] = []
    spatial: dict[str, dict] = {}
    gini_badges: list[dict] = []

    for pol in pollutants:
        segs = []
        sector_totals: dict[str, float] = {}
        for sid in sectors:
            df = ctx.grid_df(sid, "area", pol)
            total = float(df["emission"].sum()) if not df.empty else 0.0
            sector_totals[sid] = total
            sm = sector_viz_meta(sid)
            summary_rows.append({
                "sector_id": sid,
                "sector": sm.get("tree_label", sid),
                "pollutant": pol,
                "total": total,
                "total_label": _fmt_sci(total),
                "color": sm.get("accent", "#4f7cff"),
            })
        grand = sum(sector_totals.values())
        summary_rows.append({
            "sector_id": "TOTAL",
            "sector": "TOTAL",
            "pollutant": pol,
            "total": grand,
            "total_label": _fmt_sci(grand),
            "color": "#e8eaf0",
        })
        for sid, val in sector_totals.items():
            sm = sector_viz_meta(sid)
            pct = (100.0 * val / grand) if grand > 0 else 0.0
            segs.append({
                "sector_id": sid,
                "label": sm.get("tree_label", sid),
                "value": val,
                "value_label": _fmt_sci(val),
                "percent": round(pct, 2),
                "color": sm.get("accent", "#4f7cff"),
            })
        segs.sort(key=lambda x: x["value"], reverse=True)
        composition[pol] = segs

        total_r = ctx.area_raster("TOTAL", pol, None)
        vals = total_r.data.ravel()
        positive = vals[vals > 0]
        spatial[pol] = {
            "histogram": _log_histogram(positive),
            "lorenz": _lorenz_curve(positive),
            "threshold": ctx.get_threshold(pol),
            "threshold_label": _fmt_sci(ctx.get_threshold(pol)),
        }
        for sid in sectors:
            r = ctx.area_raster(sid, pol, None)
            pv = r.data[r.data > 0].ravel()
            g = _gini(pv) if pv.size else 0.0
            sm = sector_viz_meta(sid)
            gini_badges.append({
                "sector_id": sid,
                "pollutant": pol,
                "label": sm.get("tree_label", sid),
                "gini": round(g, 3),
                "color": sm.get("accent", "#4f7cff"),
            })

    pol_totals = {pol: 0.0 for pol in pollutants}
    sector_vals: dict[str, list[float]] = {}
    for sid in sectors:
        row = []
        for pol in pollutants:
            df = ctx.grid_df(sid, "area", pol)
            v = float(df["emission"].sum()) if not df.empty else 0.0
            row.append(v)
            pol_totals[pol] += v
        sector_vals[sid] = row

    for sid in sectors:
        sm = sector_viz_meta(sid)
        vals = sector_vals[sid]
        pct = [
            round(100.0 * v / pol_totals[pol], 2) if pol_totals[pol] > 0 else 0.0
            for pol, v in zip(pollutants, vals)
        ]
        radar.append({
            "sector_id": sid,
            "label": sm.get("tree_label", sid),
            "color": sm.get("accent", "#4f7cff"),
            "pollutants": pollutants,
            "values": vals,
            "values_pct": pct,
            "value_labels": [_fmt_sci(v) for v in vals],
        })

    defaults = {pol: default_threshold(pol) for pol in pollutants}
    return {
        "unit": "kg/yr/cell",
        "pollutants": pollutants,
        "sectors": sectors,
        "composition": composition,
        "summary": summary_rows,
        "radar": radar,
        "spatial": spatial,
        "gini": gini_badges,
        "default_thresholds": defaults,
    }


def viewport_stats(
    ctx: RunContext,
    pollutant: str,
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict[str, Any]:
    r = ctx.area_raster("TOTAL", pollutant, None)
    tr = Transformer.from_crs(r.crs, "EPSG:4326", always_xy=True)
    total = 0.0
    sector_sums: dict[str, float] = {sid: 0.0 for sid in _sector_area_ids(ctx)}

    h, w = r.data.shape
    for row in range(h):
        for col in range(w):
            v = float(r.data[row, col])
            if v <= 0:
                continue
            x, y = r.transform * (col + 0.5, row + 0.5)
            lon, lat = tr.transform(x, y)
            if not (west <= lon <= east and south <= lat <= north):
                continue
            total += v
            for sid in sector_sums:
                sr = ctx.area_raster(sid, pollutant, None)
                if row < sr.data.shape[0] and col < sr.data.shape[1]:
                    sv = float(sr.data[row, col])
                    if sv > 0:
                        sector_sums[sid] += sv

    dominant = None
    if sector_sums:
        dominant_id = max(sector_sums, key=sector_sums.get)
        if sector_sums[dominant_id] > 0:
            sm = sector_viz_meta(dominant_id)
            dominant = sm.get("tree_label", dominant_id)

    facility_count = 0
    from UrbEm_Visualizer.visualization.geojson_layers import points_geojson

    gj = points_geojson(ctx, _sector_area_ids(ctx), pollutant)
    for feat in gj.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        if west <= lon <= east and south <= lat <= north:
            facility_count += 1

    return {
        "pollutant": pollutant,
        "emission_total": total,
        "emission_label": _fmt_sci(total),
        "unit": "kg/yr/cell",
        "facility_count": facility_count,
        "dominant_sector": dominant or "—",
    }


def facility_comparison(
    ctx: RunContext,
    lon: float,
    lat: float,
    pollutant: str,
    sector_ids: list[str] | None = None,
) -> dict[str, Any]:
    from UrbEm_Visualizer.visualization.geojson_layers import facility_detail

    if sector_ids is None:
        sector_ids = [s for s in ctx.sector_ids() if ctx.sector_layers(s)["point"]]
    base = facility_detail(ctx, lon, lat, pollutant, sector_ids)
    if base.get("pollutants"):
        return base
    rows = []
    for row in base.get("sectors") or []:
        sid = row["sector"]
        df = ctx.grid_df(sid, "area", pollutant)
        median = float(np.median(df["emission"])) if not df.empty else 0.0
        em = float(row["emission"])
        ratio = (em / median) if median > 0 else 0.0
        rows.append({
            **row,
            "emission_label": _fmt_sci(em),
            "median": median,
            "median_label": _fmt_sci(median),
            "ratio": round(ratio, 2),
            "ratio_label": f"{ratio:.1f}× sector median" if median > 0 else "—",
        })
    base["sectors"] = rows
    return base
