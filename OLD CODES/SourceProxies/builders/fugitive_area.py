"""GNFR D fugitive area weights from CEIP + OSM + CORINE + population (CAMS-cell normalized)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import rasterio

from Waste.j_waste_weights.cams_grid import build_cam_cell_id
from Waste.j_waste_weights.normalization import validate_weight_sums

from PROXY.core.dataloaders import load_yaml
from PROXY.sectors.D_Fugitive.pipeline import run_fugitive_pipeline

from ..grid import resolve_path


def build_fugitive_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    rel = Path(str(sector_entry.get("fugitive_config", "PROXY/config/fugitive/area_source.yaml")))
    cfg_path = resolve_path(root, rel)
    fugitive_cfg = load_yaml(cfg_path)
    fugitive_cfg["_project_root"] = root

    paths = dict(fugitive_cfg.get("paths") or {})
    main_paths = cfg.get("paths") or {}
    for key in ("cams_nc", "nuts_gpkg", "corine"):
        if key in main_paths:
            paths[key] = main_paths[key]
    fugitive_cfg["paths"] = paths

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_name = str(sector_entry.get("filename", "Fugitive_areasource.tif"))
    fugitive_cfg["output"] = {
        **(fugitive_cfg.get("output") or {}),
        "dir": str(out_dir),
        "weights_tif": out_name,
    }

    fb = str((cfg.get("country") or {}).get("cams_iso3") or (fugitive_cfg.get("defaults") or {}).get("fallback_country_iso3", "GRC"))

    out_tif = run_fugitive_pipeline(
        root,
        fugitive_cfg,
        ref,
        country_iso3_fallback=fb,
        show_progress=bool(cfg.get("show_progress", True)),
    )

    if run_validate:
        nc = Path(paths["cams_nc"])
        if not nc.is_absolute():
            nc = root / nc
        cam_cell_id = build_cam_cell_id(nc, ref)
        pollutants = [str(p) for p in fugitive_cfg["pollutants"]]
        with rasterio.open(out_tif) as src:
            for i in range(1, src.count + 1):
                arr = src.read(i)
                errs = validate_weight_sums(arr, cam_cell_id, None, tol=1e-3)
                label = pollutants[i - 1] if i - 1 < len(pollutants) else f"band_{i}"
                if not errs:
                    print(f"validate: fugitive_area band {label!r} OK (CAMS-cell sums ~ 1).", file=sys.stderr)
                else:
                    print(
                        f"validate: fugitive_area band {label!r} — {len(errs)} issue(s), e.g. {errs[:3]}",
                        file=sys.stderr,
                    )

    return out_tif
