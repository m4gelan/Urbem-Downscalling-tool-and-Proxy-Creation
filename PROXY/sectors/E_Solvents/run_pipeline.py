"""End-to-end GNFR E solvent area downscaling to fine-grid pollutant weights."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.alpha import finalize_alpha_matrix
from PROXY.core.area_allocation import allocate_weights_from_normalized_stack
from PROXY.core.cams.grid import build_cams_source_index_grid
from PROXY.core.ceip import load_ceip_and_alpha_solvents
from PROXY.core.corine.raster import clc_group_masks, read_corine_window
from PROXY.core.dataloaders import resolve_path as project_resolve
from PROXY.core.logging_tables import log_wide_group_alpha_table
from PROXY.core.raster import (
    normalize_stack_within_cells,
    validate_non_negative,
    validate_parent_weight_sums_strict,
    warp_sum_to_ref,
)
from PROXY.core.ref_profile import load_area_ref_profile

from .archetypes import build_archetype_proxies, generic_proxy
from .export_weights import write_solvents_area_weights
from .osm_indicators import load_osm_indicators
from .subsector_proxies import build_subsector_raw_stack, validate_beta

logger = logging.getLogger(__name__)

_GNFR_E_INDEX = 5
_CAMS_AREA_SOURCE_INDEX = 1


def _log_raster_float(name: str, arr: np.ndarray) -> None:
    """Log shape and basic distribution for a float proxy (finite values only)."""
    a = np.asarray(arr, dtype=np.float64)
    m = np.isfinite(a)
    n = int(a.size)
    if n == 0:
        logger.info("E_Solvents raster %s: empty", name)
        return
    n_fin = int(np.count_nonzero(m))
    if n_fin == 0:
        logger.info("E_Solvents raster %s: shape=%s no finite values", name, a.shape)
        return
    sub = a[m]
    logger.info(
        "E_Solvents raster %s: shape=%s finite=%.1f%% min=%.6g p50=%.6g max=%.6g",
        name,
        a.shape,
        100.0 * n_fin / n,
        float(np.min(sub)),
        float(np.median(sub)),
        float(np.max(sub)),
    )


def load_json_config(root: Path, config_path: Path) -> dict[str, Any]:
    p = project_resolve(root, config_path)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _load_ref_profile(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    merged["_project_root"] = root
    if "corine_window" not in merged:
        cor = cfg.get("corine") or {}
        country = cfg.get("country") or {}
        merged["corine_window"] = {
            "nuts_cntr": str(country.get("nuts_cntr", "EL")),
            "pad_m": float(cor.get("pad_m", 5000.0)),
        }
    return load_area_ref_profile(merged)


def _warp_population_to_ref(root: Path, pop_path: str | Path, ref: dict[str, Any]) -> np.ndarray:
    p = project_resolve(root, Path(pop_path))
    h, w = int(ref["height"]), int(ref["width"])
    if not p.is_file():
        return np.zeros((h, w), dtype=np.float32)
    return warp_sum_to_ref(p, ref)


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

    logger.info(
        "E_Solvents pipeline start: country cams_iso3=%s subsectors=%s pollutants=%s",
        iso3,
        subsectors,
        pollutants,
    )

    cfg = dict(cfg)
    cfg["_project_root"] = root

    ref = _load_ref_profile(root, cfg)
    logger.info(
        "E_Solvents [1/9] reference grid: height=%s width=%s crs=%r domain_bbox_wgs84=%s",
        ref.get("height"),
        ref.get("width"),
        ref.get("crs"),
        ref.get("domain_bbox_wgs84"),
    )

    alpha_iso, _fb, wide, ceip_meta = load_ceip_and_alpha_solvents(
        cfg, [iso3], focus_country_iso3=iso3
    )
    logger.info(
        "E_Solvents [2/9] CEIP alpha: source=%s meta=%s",
        ceip_meta.get("source", "?"),
        {k: ceip_meta[k] for k in ("path", "ceip_years") if k in ceip_meta},
    )
    group_cols = tuple(f"alpha_{s}" for s in subsectors)
    log_wide_group_alpha_table(
        logger,
        sector="E_Solvents",
        wide=wide,
        focus_iso3=iso3,
        group_cols=group_cols,
    )

    b_errs = validate_beta(beta, ("house", "serv", "ind", "infra"))
    if b_errs:
        raise ValueError("beta validation: " + "; ".join(b_errs))
    logger.info("E_Solvents [3/9] beta / generic_omega validated (archetype weights OK)")

    nc = project_resolve(root, Path(paths["cams_nc"]))
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")
    logger.info("E_Solvents [4/9] CAMS NetCDF: %s", nc)
    cell_of, m_e = build_cams_source_index_grid(
        nc,
        ref,
        gnfr_index=_GNFR_E_INDEX,
        source_type_index=_CAMS_AREA_SOURCE_INDEX,
        country_iso3=iso3,
    )
    n_pix = int(cell_of.size)
    n_mapped = int(np.count_nonzero(cell_of >= 0))
    n_cams_e = int(np.count_nonzero(m_e))
    uniq_parents = int(len(np.unique(cell_of[cell_of >= 0]))) if n_mapped else 0
    logger.info(
        "E_Solvents [4/9] fine grid -> CAMS GNFR E area parents: pixels mapped=%d/%d (%.1f%%), "
        "unique CAMS source indices=%d, CAMS rows GNFR E area (country)=%d",
        n_mapped,
        n_pix,
        100.0 * n_mapped / max(n_pix, 1),
        uniq_parents,
        n_cams_e,
    )

    pop = _warp_population_to_ref(root, paths.get("population_tif", ""), ref)
    _log_raster_float("population", pop)
    corine_fs = project_resolve(root, Path(paths.get("corine", "")))
    h, w = int(ref["height"]), int(ref["width"])
    if corine_fs.is_file():
        clc = read_corine_window(root, cfg, ref)
        logger.info("E_Solvents [5/9] CORINE window: path=%s clc dtype=%s", corine_fs, clc.dtype)
    else:
        warnings.warn(
            f"CORINE raster not found ({corine_fs.as_posix()}); CLC-based masks are zero.",
            stacklevel=1,
        )
        clc = np.full((h, w), -9999, dtype=np.int32)
        logger.warning("E_Solvents [5/9] CORINE missing — using sentinel grid shape=%s", clc.shape)
    masks = clc_group_masks(clc, cfg.get("corine_codes") or {})
    ind_clc = masks.get("industrial_clc")
    if ind_clc is None:
        ind_clc = np.zeros_like(pop, dtype=np.float32)
    for mk in ("residential_share", "urban_fabric", "service_land", "industrial_clc"):
        if mk in masks:
            _log_raster_float(f"CLC mask {mk}", masks[mk])
    logger.info("E_Solvents [5/9] CLC mask keys built: %s", sorted(masks.keys()))

    raw: dict[str, np.ndarray] = {
        "population": pop,
        "residential_share": masks.get("residential_share", np.zeros_like(pop)),
        "urban_fabric": masks.get("urban_fabric", np.zeros_like(pop)),
        "service_land": masks.get("service_land", np.zeros_like(pop)),
        "industry_clc": ind_clc,
    }
    raw.update(load_osm_indicators(root, cfg, ref))
    logger.info("E_Solvents [6/9] raw proxy keys: %s", sorted(raw.keys()))
    for k in sorted(raw.keys()):
        _log_raster_float(f"raw[{k}]", raw[k])

    arche = build_archetype_proxies(raw, cfg)
    logger.info("E_Solvents [7/9] archetype proxy keys: %s", sorted(arche.keys()))
    for k in sorted(arche.keys()):
        _log_raster_float(f"archetype[{k}]", arche[k])

    gen = generic_proxy(arche, omega)
    _log_raster_float("generic_proxy", gen)

    rho_raw = build_subsector_raw_stack(arche, beta, subsectors)
    logger.info(
        "E_Solvents [8/9] subsector raw stack rho_raw: shape=%s dtype=%s",
        rho_raw.shape,
        rho_raw.dtype,
    )
    rho_norm, fb_log = normalize_stack_within_cells(cell_of, rho_raw, gen)
    logger.info(
        "E_Solvents [8/9] within-CAMS normalize: rho_norm shape=%s fallback_events=%d (capped at 5000 in meta)",
        rho_norm.shape,
        len(fb_log),
    )

    alpha = np.asarray(alpha_iso[0], dtype=np.float64).T
    alpha = finalize_alpha_matrix(alpha, subsectors)
    logger.info(
        "E_Solvents [9/9] allocate weights: alpha shape=%s pollutants=%s",
        alpha.shape,
        pollutants,
    )
    W = allocate_weights_from_normalized_stack(rho_norm, alpha)
    logger.info("E_Solvents [9/9] output W shape=%s dtype=%s", W.shape, W.dtype)

    tol = float(cfg.get("tolerance_mass", 1e-5))
    v1 = validate_parent_weight_sums_strict(W, cell_of, tol=tol)
    v2 = validate_non_negative(W)
    if v1 or v2:
        raise ValueError("Output validation: " + "; ".join(v1 + v2))
    logger.info(
        "E_Solvents validation OK: within_cell_mass tol=%.3g (no errors), non_negative OK",
        tol,
    )

    out_dir = project_resolve(root, Path(cfg.get("output_dir", "OUTPUT/Proxy_weights/E_Solvents")))
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
    logger.info("E_Solvents pipeline complete: wrote %s", out_tif)
    return {"output_tif": str(out_tif), "meta": meta}
