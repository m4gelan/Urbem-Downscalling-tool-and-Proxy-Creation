from __future__ import annotations

from pathlib import Path

import yaml

from UrbEm_Visualizer.paths import package_dir

_CFG_PATH = package_dir() / "config" / "map_visualization.yaml"
_CACHE: dict | None = None


def load_map_config() -> dict:
    global _CACHE
    if _CACHE is None:
        with open(_CFG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{_CFG_PATH}: invalid map config")
        _CACHE = data
    return _CACHE


def sector_viz_meta(sector_id: str) -> dict:
    cfg = load_map_config()
    sectors = cfg.get("sectors") or {}
    base = {
        "id": sector_id,
        "tree_label": sector_id.replace("_", ""),
        "accent": "#4f7cff",
        "icon": "dot",
    }
    if sector_id in sectors:
        base.update(sectors[sector_id])
    return base
