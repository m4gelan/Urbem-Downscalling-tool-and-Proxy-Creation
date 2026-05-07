"""GNFR B Industry area-source weights (OSM groups + CORINE + CEIP alphas)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import rasterio

from SourceProxies.grid import resolve_path


def build_industry_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    import yaml

    from Industry.industry_areasource import (
        CAMS_SOURCE_TYPE_AREA,
        GNFR_B_INDEX_DEFAULT,
        run_from_project_config,
    )
    from Waste.j_waste_weights.cams_grid import (
        build_cam_cell_id,
        build_cam_cell_id_masked_for_j_sources,
    )
    from Waste.j_waste_weights.normalization import validate_weight_sums

    rel = Path(str(sector_entry.get("industry_config", "Industry/config/industry_area.yaml")))
    ypath = rel if rel.is_absolute() else (root / rel)
    if not ypath.is_file():
        raise FileNotFoundError(f"Industry area config not found: {ypath}")

    with ypath.open(encoding="utf-8") as f:
        icfg = yaml.safe_load(f)

    paths = dict(icfg.get("paths") or {})
    main_paths = cfg.get("paths") or {}
    for key in (
        "cams_nc",
        "nuts_gpkg",
        "corine",
        "population_tif",
        "pop_path",
        "ceip_industry",
        "osm_G1",
        "osm_G2",
        "osm_G3",
        "osm_G4",
    ):
        if key in main_paths:
            paths[key] = main_paths[key]
    icfg["paths"] = paths

    country = cfg.get("country") or {}
    if country:
        icfg["country"] = country

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    icfg.setdefault("output", {})
    icfg["output"]["dir"] = str(out_dir)
    icfg["output"]["weights_tif"] = str(sector_entry.get("filename", "Industry_areasource.tif"))

    out_tif = run_from_project_config(icfg, root)

    if run_validate:
        industry = icfg.get("industry") or {}
        use_mask = bool(industry.get("use_cams_area_mask", True))
        gnfr = int(industry.get("gnfr_b_index", GNFR_B_INDEX_DEFAULT))
        iso_fb = str(
            industry.get("fallback_country_iso3")
            or (icfg.get("defaults") or {}).get("fallback_country_iso3")
            or "GRC"
        ).strip().upper()
        nc = resolve_path(root, Path(paths["cams_nc"]))
        if use_mask:
            cam_cell_id = build_cam_cell_id_masked_for_j_sources(
                nc,
                ref,
                gnfr_j_index=gnfr,
                source_type=CAMS_SOURCE_TYPE_AREA,
                country_iso3=iso_fb,
            )
        else:
            cam_cell_id = build_cam_cell_id(nc, ref)
        pollutants = [str(p) for p in (icfg.get("industry") or {}).get("pollutants", ["NOx"])]
        with rasterio.open(out_tif) as src:
            for i in range(1, src.count + 1):
                arr = src.read(i)
                errs = validate_weight_sums(arr, cam_cell_id, None, tol=1e-3)
                label = pollutants[i - 1] if i - 1 < len(pollutants) else f"band_{i}"
                if not errs:
                    print(f"validate: industry_area band {label!r} OK (CAMS-cell sums ~ 1).", file=sys.stderr)
                else:
                    print(
                        f"validate: industry_area band {label!r} — {len(errs)} issue(s), e.g. {errs[:3]}",
                        file=sys.stderr,
                    )

    return out_tif
