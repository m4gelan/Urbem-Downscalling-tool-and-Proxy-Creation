"""End-to-end GNFR E solvent area downscaling to fine-grid pollutant weights."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..indicators.corine import clc_group_masks, read_corine_window
from ..indicators.osm_pbf import load_osm_indicators
from ..indicators.population import warp_population_to_ref
from ..io.ceip import load_ceip_alpha_table, validate_alpha
from ..io.export_weights import write_solvents_area_weights
from ..io.paths import resolve_path as project_resolve
from ..io.ref_grid import load_ref_profile
from .allocate import allocate_weights, alpha_matrix, finalize_alpha_matrix
from .archetypes import build_archetype_proxies, generic_proxy
from .cams_fine_grid import build_cell_of
from .subsector_proxies import build_subsector_raw_stack, validate_beta
from .validate import check_non_negative, check_within_cell_mass
from .within_cams import normalize_within_cams_parents


def load_json_config(root: Path, config_path: Path) -> dict[str, Any]:
    p = project_resolve(root, config_path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def run_solvents_area_pipeline(
    root: Path,
    cfg: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build E_solvents_areasource.tif (or cfg output_tif) under output_dir.

    Returns dict with paths and diagnostics.
    """
    paths = cfg["paths"]
    country = cfg["country"]
    iso3 = str(country["cams_iso3"]).strip().upper()
    subsectors: list[str] = list(cfg["subsectors"])
    pollutants: list[str] = list(cfg["pollutants"])
    beta = cfg.get("beta") or {}
    omega = cfg.get("generic_omega") or {}

    ref = load_ref_profile(root, cfg)
    agg, ceip_meta = load_ceip_alpha_table(root, cfg, target_iso3=iso3)
    a_errs = validate_alpha(agg, subsectors)
    if a_errs:
        raise ValueError("CEIP alpha validation: " + "; ".join(a_errs))

    b_errs = validate_beta(beta, ("house", "serv", "ind", "infra"))
    if b_errs:
        raise ValueError("beta validation: " + "; ".join(b_errs))

    nc = project_resolve(root, Path(paths["cams_nc"]))
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")
    cell_of, _m_e = build_cell_of(nc, iso3, ref)

    pop = warp_population_to_ref(root, paths.get("population_tif", ""), ref)
    corine_fs = project_resolve(root, Path(paths.get("corine", "")))
    h, w = int(ref["height"]), int(ref["width"])
    if corine_fs.is_file():
        clc = read_corine_window(root, cfg, ref)
    else:
        warnings.warn(
            f"CORINE raster not found ({corine_fs.as_posix()}); CLC-based masks are zero.",
            stacklevel=1,
        )
        clc = np.full((h, w), -9999, dtype=np.int32)
    masks = clc_group_masks(clc, cfg.get("corine_codes") or {})
    ind_clc = masks.get("industrial_clc")
    if ind_clc is None:
        ind_clc = np.zeros_like(pop, dtype=np.float32)

    raw: dict[str, np.ndarray] = {
        "population": pop,
        "residential_share": masks.get("residential_share", np.zeros_like(pop)),
        "urban_fabric": masks.get("urban_fabric", np.zeros_like(pop)),
        "service_land": masks.get("service_land", np.zeros_like(pop)),
        "industry_clc": ind_clc,
    }
    raw.update(load_osm_indicators(root, cfg, ref))

    arche = build_archetype_proxies(raw, cfg)
    gen = generic_proxy(arche, omega)
    rho_raw = build_subsector_raw_stack(arche, beta, subsectors)
    rho_norm, fb_log = normalize_within_cams_parents(cell_of, rho_raw, gen)

    alpha = alpha_matrix(agg, pollutants, subsectors)
    alpha = finalize_alpha_matrix(alpha, subsectors)
    W = allocate_weights(rho_norm, alpha)

    tol = float(cfg.get("tolerance_mass", 1e-5))
    v1 = check_within_cell_mass(W, cell_of, tol=tol)
    v2 = check_non_negative(W)
    if v1 or v2:
        raise ValueError("Output validation: " + "; ".join(v1 + v2))

    out_dir = project_resolve(root, Path(cfg.get("output_dir", "Solvents/outputs")))
    out_name = str(cfg.get("output_tif", "E_solvents_areasource.tif"))
    out_tif = out_dir / out_name

    meta: dict[str, Any] = {
        "ceip": ceip_meta,
        "country_iso3": iso3,
        "pollutants": pollutants,
        "subsectors": subsectors,
        "within_cams_fallback_events": fb_log[:5000],
        "within_cams_fallback_truncated": len(fb_log) > 5000,
        "ref": {
            "height": ref["height"],
            "width": ref["width"],
            "crs": ref["crs"],
            "domain_bbox_wgs84": list(ref["domain_bbox_wgs84"]),
        },
        "config_path": str(config_path) if config_path else None,
    }
    write_solvents_area_weights(out_tif, W, ref, pollutants, meta=meta)
    return {"output_tif": str(out_tif), "meta": meta}
