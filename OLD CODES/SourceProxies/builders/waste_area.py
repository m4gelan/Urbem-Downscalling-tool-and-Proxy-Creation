"""GNFR J waste within-cell weights (area + point CAMS streams) from CEIP + proxies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Waste.j_waste_weights.config_loader import load_config
from Waste.j_waste_weights.main import run_with_config


def build_waste_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    _ = ref
    _ = run_validate
    rel = Path(str(sector_entry.get("waste_config", "Waste/j_waste_weights/config.yaml")))
    cfg_path = rel if rel.is_absolute() else (root / rel)
    wcfg = load_config(cfg_path)
    wcfg["_project_root"] = root

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    area_name = str(sector_entry.get("filename", "Waste_sourcearea.tif"))
    point_name = str(sector_entry.get("point_filename", "Waste_pointarea.tif"))

    paths = dict(wcfg.get("paths") or {})
    main_paths = cfg.get("paths") or {}
    for key in ("cams_nc", "nuts_gpkg", "corine"):
        if key in main_paths:
            paths[key] = main_paths[key]
    wcfg["paths"] = paths

    wcfg["output"] = {
        **(wcfg.get("output") or {}),
        "dir": str(out_dir),
        "weights_tif_area": area_name,
        "weights_tif_point": point_name,
    }

    rc = run_with_config(wcfg)
    if rc != 0:
        raise SystemExit(f"waste_j pipeline failed with exit code {rc}")
    return out_dir / area_name
