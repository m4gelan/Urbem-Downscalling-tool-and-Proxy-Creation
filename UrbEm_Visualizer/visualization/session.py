from __future__ import annotations

from pathlib import Path
from typing import Any

from UrbEm_Visualizer.visualization.emission_style import default_threshold
from UrbEm_Visualizer.visualization.load_run import RunContext, domain_wgs84, load_manifest
from UrbEm_Visualizer.visualization.map_config import load_map_config, sector_viz_meta
from UrbEm_Visualizer.visualization.validate import validate_output_folder

_VIZ: dict[str, Any] = {"context": None, "output_dir": None}


def open_output(output_dir: Path) -> RunContext:
    config = load_manifest(output_dir)
    ctx = RunContext(output_dir, config)
    _VIZ["context"] = ctx
    _VIZ["output_dir"] = str(output_dir.resolve())
    return ctx


def get_context() -> RunContext | None:
    return _VIZ.get("context")


def build_meta(ctx: RunContext) -> dict[str, Any]:
    w, s, e, n = domain_wgs84(ctx.domain)
    sectors = []
    for sid in ctx.sector_ids():
        layers = ctx.sector_layers(sid)
        meta = sector_viz_meta(sid)
        sectors.append({
            "id": sid,
            "label": meta.get("tree_label", sid),
            "accent": meta.get("accent", "#4f7cff"),
            "icon": meta.get("icon", "dot"),
            "has_area": layers["area"],
            "has_point": layers["point"],
        })
    return {
        "output_dir": str(ctx.output_dir),
        "pollutants": ctx.pollutants,
        "unit": ctx.unit,
        "layer_mode": ctx.layer_mode,
        "domain_wgs84": {"west": w, "south": s, "east": e, "north": n},
        "domain_crs": str(ctx.domain.get("crs", "")),
        "sectors": sectors,
        "map_config": load_map_config(),
        "sector_scale": ctx.sector_scale,
        "per_sector_scale": ctx.per_sector_scale,
        "total_scale": ctx.total_scale,
        "default_thresholds": {pol: default_threshold(pol) for pol in ctx.pollutants},
    }
