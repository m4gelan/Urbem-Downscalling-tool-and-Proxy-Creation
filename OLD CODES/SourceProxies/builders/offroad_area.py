"""GNFR I off-road area weights (rail + pipeline + non-road + CEIP shares, CAMS-cell normalized)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import rasterio
import yaml

from Waste.j_waste_weights.cams_grid import build_cam_cell_id
from Waste.j_waste_weights.normalization import validate_weight_sums


def build_offroad_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    """
    Build ``Offroad_Sourcearea.tif`` on the same CORINE+NUTS reference grid as other sectors.

    YAML defaults: ``Offroad/config/offroad_area.yaml`` (override via ``sector_entry["offroad_config"]``).
    """
    rel = Path(
        str(
            sector_entry.get("offroad_config")
            or "Offroad/config/offroad_area.yaml"
        )
    )
    cfg_path = rel if rel.is_absolute() else (root / rel)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Offroad config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        off_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    paths = dict(off_cfg.get("paths") or {})
    main_paths = cfg.get("paths") or {}
    for key in ("cams_nc", "corine", "nuts_gpkg"):
        if key in main_paths:
            paths[key] = main_paths[key]
    if main_paths.get("landscan"):
        paths["population_tif"] = main_paths["landscan"]

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_name = str(
        sector_entry.get("filename")
        or (off_cfg.get("output") or {}).get("filename")
        or "Offroad_Sourcearea.tif"
    )
    output_tif = out_dir / out_name

    pollutants = sector_entry.get("pollutants")
    if pollutants is not None:
        pollutants = [str(x) for x in pollutants]

    from Offroad.offroad_areasource import run_offroad_areasource

    out_tif = run_offroad_areasource(
        root=root,
        ref=ref,
        yaml_cfg=off_cfg,
        paths=paths,
        output_tif=output_tif,
        pollutants=pollutants,
    )

    if run_validate:
        import numpy as np

        cams_nc = paths["cams_nc"]
        cams_p = Path(cams_nc)
        if not cams_p.is_absolute():
            cams_p = root / cams_p
        cam = build_cam_cell_id(cams_p, ref).astype(np.int64)
        with rasterio.open(out_tif) as src:
            for b in range(1, int(src.count) + 1):
                arr = src.read(b)
                errs = validate_weight_sums(arr, cam, None, tol=1e-3)
                if not errs:
                    print(
                        f"validate: offroad_area band {b} OK (CAMS-cell sums ~ 1).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"validate: offroad_area band {b} — {len(errs)} issue(s), e.g. {errs[:3]}",
                        file=sys.stderr,
                    )

    return Path(out_tif)
