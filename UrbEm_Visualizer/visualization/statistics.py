from __future__ import annotations

from typing import Any

import numpy as np

from UrbEm_Visualizer.visualization.load_run import RunContext
from UrbEm_Visualizer.visualization.map_config import sector_viz_meta


def compute_statistics(
    ctx: RunContext,
    pollutant: str,
    active_sectors: list[str],
) -> dict[str, Any]:
    summary = []
    sector_totals: dict[str, float] = {}
    hist_vals: list[float] = []

    for sid in active_sectors:
        layers = ctx.sector_layers(sid)
        total = 0.0
        if layers["area"]:
            df = ctx.grid_df(sid, "area", pollutant)
            total += float(df["emission"].sum()) if not df.empty else 0.0
            if not df.empty:
                hist_vals.extend(df["emission"].astype(float).tolist())
        if layers["point"]:
            df = ctx.grid_df(sid, "point", pollutant)
            total += float(df["emission"].sum()) if not df.empty else 0.0
        sector_totals[sid] = total
        meta = sector_viz_meta(sid)
        summary.append({
            "sector_id": sid,
            "sector": meta.get("tree_label", sid),
            "pollutant": pollutant,
            "total": round(total, 4),
            "color": meta.get("accent", "#4f7cff"),
        })

    grand = sum(sector_totals.values())
    summary.append({
        "sector_id": "TOTAL",
        "sector": "TOTAL",
        "pollutant": pollutant,
        "total": round(grand, 4),
        "color": "#e8eaf0",
    })

    bars = []
    for sid, val in sector_totals.items():
        pct = (100.0 * val / grand) if grand > 0 else 0.0
        meta = sector_viz_meta(sid)
        bars.append({
            "sector_id": sid,
            "label": meta.get("tree_label", sid),
            "value": round(val, 4),
            "percent": round(pct, 2),
            "color": meta.get("accent", "#4f7cff"),
        })
    bars.sort(key=lambda x: x["value"], reverse=True)

    pie = [{"sector_id": b["sector_id"], "label": b["label"], "value": b["value"], "color": b["color"]} for b in bars]

    histogram = _histogram(hist_vals, bins=24)

    return {
        "unit": ctx.unit,
        "pollutant": pollutant,
        "summary": summary,
        "bars": bars,
        "pie": pie,
        "histogram": histogram,
        "bbox": {
            "crs": ctx.domain["crs"],
            "xmin": ctx.domain["xmin"],
            "ymin": ctx.domain["ymin"],
            "xmax": ctx.domain["xmax"],
            "ymax": ctx.domain["ymax"],
        },
    }


def _histogram(values: list[float], bins: int = 24) -> dict[str, Any]:
    if not values:
        return {"labels": [], "counts": []}
    arr = np.asarray(values, dtype=np.float32)
    counts, edges = np.histogram(arr, bins=bins)
    labels = [f"{edges[i]:.2g}–{edges[i + 1]:.2g}" for i in range(len(counts))]
    return {"labels": labels, "counts": [int(c) for c in counts]}
