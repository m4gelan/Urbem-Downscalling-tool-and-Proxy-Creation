"""Per-layer CORINE + OSM masks for Solvents maps (same ingredients as the pipeline, ref window)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io.paths import resolve_path as project_resolve
from .corine import clc_group_masks, read_corine_window
from .osm_pbf import (
    OSM_SOLVENT_FAMILY_COVER,
    OSM_SOLVENT_LAYER_COVER,
    rasterize_solvent_gpkg_by_family,
)


def build_solvent_context_masks(root: Path, cfg: dict[str, Any], ref: dict) -> dict[str, np.ndarray]:
    """
    Binary float32 masks on ``ref``: ``corine_<group>`` for each ``corine_codes`` entry,
    and ``osm_<layer>`` / ``osm_<solvent_family>`` when ``paths.osm_solvent_gpkg`` is available.
    """
    out: dict[str, np.ndarray] = {}
    corine_path = project_resolve(root, Path(cfg.get("paths", {}).get("corine", "")))
    if corine_path.is_file():
        clc = read_corine_window(root, cfg, ref)
        masks = clc_group_masks(clc, cfg.get("corine_codes") or {})
        for name, m in sorted(masks.items()):
            out[f"corine_{name}"] = np.asarray(m, dtype=np.float32)
    raw = (cfg.get("paths") or {}).get("osm_solvent_gpkg")
    if raw:
        p = project_resolve(root, Path(raw))
        if p.is_file():
            fam = rasterize_solvent_gpkg_by_family(root, cfg, ref) or {}
            rl = fam.pop("_roads_rl", None)
            for k in sorted(fam.keys()):
                if str(k).startswith("_"):
                    continue
                layer_key = k if str(k).startswith("osm_") else f"osm_{k}"
                out[layer_key] = np.asarray(fam[k], dtype=np.float32)
            if rl is not None:
                out["osm_roads"] = (np.asarray(rl, dtype=np.float32) > 0).astype(np.float32)
    return out


def title_for_solvent_mask_key(key: str) -> str:
    if key.startswith("corine_"):
        name = key[7:]
        return f"Solvents · CORINE · {name} · binary"
    if key.startswith("osm_"):
        tail = key[4:]
        desc = OSM_SOLVENT_LAYER_COVER.get(key) or OSM_SOLVENT_FAMILY_COVER.get(tail)
        if desc is None:
            desc = tail
        return f"Solvents · OSM · {tail} ({desc}) · binary"
    return f"Solvents · {key}"
