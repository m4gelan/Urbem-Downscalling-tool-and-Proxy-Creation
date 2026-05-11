"""Merge PROXY path config + sector YAML and run the J_Waste pipeline."""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.cams import build_cam_cell_id_masked_for_sources
from PROXY.core.alpha import default_ceip_profile_relpath, load_ceip_and_alpha
from PROXY.core.dataloaders import load_first_existing_yaml_or_json, resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.grid import resolve_nuts_cntr_code
from PROXY.core.logging_tables import log_waste_family_weights
from PROXY.core.raster import (
    cntr_code_to_iso3,
    normalize_within_cams_cells,
    rasterize_country_ids,
    validate_weight_sums,
)
from PROXY.core.ref_profile import load_area_ref_profile
from PROXY.sectors.J_Waste import composite_waste, diagnostics_waste
from PROXY.sectors.J_Waste.proxy_waste import build_all_proxies

logger = logging.getLogger(__name__)


def _load_waste_base(root: Path) -> dict[str, Any]:
    candidates = [
        root / "PROXY" / "config" / "ceip" / "profiles" / "J_Waste_rules.yaml",
        root / "PROXY" / "config" / "ceip" / "profiles" / "waste_pipeline.yaml",
        root / "PROXY" / "config" / "waste" / "defaults.json",
    ]
    return load_first_existing_yaml_or_json(
        candidates,
        context=(
            "J_Waste base config (expected PROXY/config/ceip/profiles/J_Waste_rules.yaml "
            "or legacy PROXY/config/ceip/profiles/waste_pipeline.yaml or "
            "PROXY/config/waste/defaults.json)"
        ),
    )


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_waste_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Build cfg for :func:`run_waste_pipeline`.

    CEIP α uses :func:`PROXY.core.alpha.load_ceip_and_alpha` (same reported workbook +
    ``ceip_groups_yaml`` / optional ``ceip_rules_yaml`` pattern as GNFR D/Fugitive),
    with ``group_order`` ``G1``–``G3``. Paths default from ``ceip/index.yaml`` via
    ``default_ceip_profile_relpath``.
    """
    cfg = deepcopy(_load_waste_base(root))
    wov = sector_cfg.get("waste") or {}
    ceip_ov = wov.get("ceip") if isinstance(wov.get("ceip"), dict) else {}
    pcommon = path_cfg.get("proxy_common") or {}
    psw = (path_cfg.get("proxy_specific") or {}).get("waste") or {}

    cams_nc = discover_cams_emissions(
        root, resolve_path(root, Path(path_cfg["emissions"]["cams_2019_nc"]))
    )
    corine = discover_corine(
        root, resolve_path(root, Path(pcommon["corine_tif"]))
    )
    nuts_gpkg = resolve_path(root, pcommon["nuts_gpkg"])
    pop_tif = resolve_path(root, pcommon["population_tif"])
    osm_waste = resolve_path(root, path_cfg["osm"]["waste"])

    alpha_w = (
        ceip_ov.get("workbook")
        or ceip_ov.get("ceip_workbook")
        or wov.get("ceip_workbook")
        or pcommon.get("alpha_workbook")
        or "INPUT/Proxy/Alpha/Reported_Emissions_EU27_2018_2023.xlsx"
    )
    ceip_workbook = resolve_path(root, Path(str(alpha_w)))

    ceip_xlsx_rel = (
        ceip_ov.get("xlsx")
        or ceip_ov.get("ceip_xlsx")
        or wov.get("ceip_xlsx")
        or "INPUT/Proxy/CEIP/CEIP_Waste.xlsx"
    )
    ceip_xlsx = resolve_path(root, Path(str(ceip_xlsx_rel)))

    imp_path = psw.get("impervious_tif")
    if not imp_path:
        raise KeyError("paths.yaml proxy_specific.waste.impervious_tif is required for J_Waste")
    imperv = resolve_path(root, Path(str(imp_path)))

    ghsl_rel = wov.get("ghsl_smod_tif") or psw.get("ghsl_smod_tif")
    if not ghsl_rel:
        raise KeyError("Set proxy_specific.waste.ghsl_smod_tif in paths.yaml (or sector waste.ghsl_smod_tif)")
    ghsl_smod = resolve_path(root, Path(str(ghsl_rel)))

    uww_agg = resolve_path(root, Path(str(psw["agglomerations_gpkg"])))
    uww_plant = resolve_path(root, Path(str(psw["treatment_plants_gpkg"])))

    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    nuts_override = str(sector_cfg.get("nuts_cntr", "")).strip().upper()
    nuts_cntr = nuts_override if len(nuts_override) == 2 else resolve_nuts_cntr_code(country)

    gnfr_idx = int((sector_cfg.get("cams_emission_category_indices") or [13])[0])

    out_base = output_dir
    w_area = wov.get("output_filename_area") or sector_cfg.get("output_filename_area")
    w_point = wov.get("output_filename_point") or sector_cfg.get("output_filename_point")
    if not w_area:
        w_area = (cfg.get("output") or {}).get("weights_tif_area", "waste_areasource.tif")
    if not w_point:
        w_point = (cfg.get("output") or {}).get("weights_tif_point", "waste_pointsource.tif")

    paths: dict[str, Any] = {
        "cams_nc": str(cams_nc.resolve()),
        "ceip_workbook": str(ceip_workbook.resolve()),
        "ceip_xlsx": str(ceip_xlsx.resolve()) if ceip_xlsx.is_file() else "",
        "ceip_families_yaml": default_ceip_profile_relpath(
            root, "J_Waste", "families_yaml"
        ),
        "nuts_gpkg": str(nuts_gpkg.resolve()),
        "corine": str(corine.resolve()),
        "population_tif": str(pop_tif.resolve()),
        "ghsl_smod_tif": str(ghsl_smod.resolve()),
        "imperviousness": str(imperv.resolve()),
        "uwwtd_agglomerations_gpkg": str(uww_agg.resolve()),
        "uwwtd_plants_gpkg": str(uww_plant.resolve()),
        "osm_waste_gpkg": str(osm_waste.resolve()),
    }

    ref_tif = wov.get("ref_tif", sector_cfg.get("ref_tif"))
    if ref_tif:
        p = resolve_path(root, Path(str(ref_tif)))
        if p.is_file():
            paths["ref_tif"] = str(p.resolve())
    psm = wov.get("point_source_mask_tif", sector_cfg.get("point_source_mask_tif"))
    if psm:
        pm = resolve_path(root, Path(str(psm)))
        if pm.is_file():
            paths["point_source_mask_tif"] = str(pm.resolve())

    ceip_sheet = ceip_ov.get("sheet") or ceip_ov.get("ceip_sheet") or wov.get("ceip_sheet")
    if ceip_sheet is not None:
        paths["ceip_sheet"] = ceip_sheet
    elif (cfg.get("paths") or {}).get("ceip_sheet") is not None:
        paths["ceip_sheet"] = (cfg.get("paths") or {}).get("ceip_sheet")
    ceip_year = ceip_ov.get("year") or ceip_ov.get("ceip_year") or wov.get("ceip_year")
    if ceip_year is not None:
        paths["ceip_year"] = ceip_year

    po = wov.get("paths")
    if isinstance(po, dict):
        for key, val in po.items():
            if val is None or str(val).strip() == "":
                continue
            paths[key] = str(resolve_path(root, Path(str(val))).resolve())

    if not str(paths.get("ceip_groups_yaml") or "").strip():
        paths["ceip_groups_yaml"] = paths["ceip_families_yaml"]
    if not str(paths.get("ceip_rules_yaml") or "").strip():
        rr = default_ceip_profile_relpath(root, "J_Waste", "rules_yaml")
        rp = resolve_path(root, Path(rr))
        if rp.is_file():
            paths["ceip_rules_yaml"] = str(rp.resolve())

    # Canonical overlay supports nested ``waste.ceip`` while keeping legacy flat keys.
    sp_merge = {
        k: v
        for k, v in wov.items()
        if k
        not in (
            "paths",
            "ceip",
            "output_filename_area",
            "output_filename_point",
        )
    }
    cfg = _deep_merge(cfg, {k: v for k, v in sp_merge.items() if v is not None})
    cfg["ceip"] = _deep_merge(cfg.get("ceip") or {}, ceip_ov)
    aliases = sector_cfg.get("ceip_pollutant_aliases")
    if isinstance(aliases, dict):
        cfg["ceip"]["pollutant_aliases"] = _deep_merge(
            cfg["ceip"].get("pollutant_aliases") or {},
            aliases,
        )
    if ceip_ov.get("years") is not None and cfg["ceip"].get("ceip_years") is None:
        cfg["ceip"]["ceip_years"] = ceip_ov.get("years")
    if not str(paths.get("alpha_method_audit_dir") or "").strip():
        paths["alpha_method_audit_dir"] = str(out_base.resolve())

    ceip_block = cfg.get("ceip") or {}
    if ceip_block.get("ceip_years") is not None:
        paths["ceip_years"] = ceip_block["ceip_years"]

    cfg["paths"] = paths
    cfg["corine_window"] = {
        "nuts_cntr": nuts_cntr,
        "pad_m": float(sector_cfg.get("pad_m", 5000.0)),
    }
    cfg["cams"] = cfg.get("cams") or {}
    cfg["cams"]["gnfr_j_index"] = gnfr_idx
    cfg["cams"]["country_iso3"] = iso3

    raw_go = (
        wov.get("ceip_group_order")
        or wov.get("group_order")
        or sector_cfg.get("ceip_group_order")
        or sector_cfg.get("group_order")
    )
    if raw_go:
        cfg["group_order"] = tuple(str(x).strip() for x in raw_go if str(x).strip())
    else:
        cfg["group_order"] = ("G1", "G2", "G3")

    poll_nc = (cfg.get("cams") or {}).get("pollutants_nc") or []
    cfg["pollutants"] = [str(p).lower().replace(".", "_") for p in poll_nc]

    cfg["ceip_pollutant_aliases"] = dict(ceip_block.get("pollutant_aliases") or {})

    cc: dict[str, Any] = {}
    for src in (ceip_ov.get("cntr_code_to_iso3"), sector_cfg.get("cntr_code_to_iso3")):
        if isinstance(src, dict):
            cc.update(src)
    cfg["cntr_code_to_iso3"] = cc

    cfg["output"] = {
        **(cfg.get("output") or {}),
        "dir": str(out_base.resolve()),
        "weights_tif_area": str(w_area),
        "weights_tif_point": str(w_point),
        "write_intermediates": bool(
            wov.get("write_intermediates", (cfg.get("output") or {}).get("write_intermediates", False))
        ),
    }
    return cfg


def _setup_logging(level: str) -> None:
    """Match :func:`PROXY.main._configure_logging` — avoid ``basicConfig`` when root is already configured."""
    try:
        log_level = getattr(logging, str(level).upper(), logging.INFO)
    except AttributeError:
        log_level = logging.INFO
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(log_level)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(log_level)


def _cams_country_iso3(cfg: dict[str, Any]) -> str:
    cams = cfg.get("cams") or {}
    raw = cams.get("country_iso3")
    if raw:
        return str(raw).strip().upper()
    cntr = (cfg.get("corine_window") or {}).get("nuts_cntr", "EL")
    iso = cntr_code_to_iso3(str(cntr))
    return iso or "GRC"


def _output_area_point_names(cfg: dict[str, Any]) -> tuple[str, str]:
    o = cfg.get("output") or {}
    area = str(o.get("weights_tif_area") or o.get("weights_tif") or "waste_areasource.tif")
    point = str(o.get("weights_tif_point") or "waste_pointsource.tif")
    return area, point


def _log_raster_stats(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if not np.any(finite):
        logger.info("J_Waste %s: shape=%s dtype=%s no finite values", name, a.shape, a.dtype)
        return
    vals = a[finite]
    logger.info(
        "J_Waste %s: shape=%s dtype=%s min=%.6g mean=%.6g max=%.6g",
        name,
        a.shape,
        a.dtype,
        float(np.min(vals)),
        float(np.mean(vals)),
        float(np.max(vals)),
    )


def run_waste_pipeline(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    c = {**cfg, "_project_root": root}
    return run_waste_j_pipeline(c)


def run_waste_j_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    log_level = (cfg.get("logging") or {}).get("level", "INFO")
    _setup_logging(str(log_level))

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = load_area_ref_profile(cfg)
    nc_path = Path(cfg["paths"]["cams_nc"])
    if not nc_path.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc_path}")

    paths = cfg.get("paths") or {}
    for key in (
        "cams_nc",
        "ceip_workbook",
        "nuts_gpkg",
        "corine",
        "population_tif",
        "ghsl_smod_tif",
        "imperviousness",
        "uwwtd_agglomerations_gpkg",
        "uwwtd_plants_gpkg",
        "osm_waste_gpkg",
    ):
        if paths.get(key):
            logger.info("J_Waste path %s=%s", key, paths[key])

    nuts_path = Path(cfg["paths"]["nuts_gpkg"])

    gnfr_j = int((cfg.get("cams") or {}).get("gnfr_j_index", 13))
    iso3_cam = _cams_country_iso3(cfg)
    cam_cell_id_area = build_cam_cell_id_masked_for_sources(
        nc_path,
        ref,
        gnfr_index=gnfr_j,
        source_type_index=1,
        country_iso3=iso3_cam,
    )
    cam_cell_id_point = build_cam_cell_id_masked_for_sources(
        nc_path,
        ref,
        gnfr_index=gnfr_j,
        source_type_index=2,
        country_iso3=iso3_cam,
    )
    logger.info(
        "J_Waste CAMS cell masks (GNFR J): area pixels %s, point pixels %s (country %s)",
        int(np.count_nonzero(cam_cell_id_area >= 0)),
        int(np.count_nonzero(cam_cell_id_point >= 0)),
        iso3_cam,
    )

    country_id, iso3_list = rasterize_country_ids(nuts_path, ref)
    logger.info(
        "J_Waste country raster: %d countries, %d pixels with ISO3",
        max(len(iso3_list) - 1, 0),
        int(np.count_nonzero(country_id > 0)),
    )

    alpha, _fb, wide = load_ceip_and_alpha(
        cfg,
        iso3_list,
        sector_key="J_Waste",
        focus_country_iso3=iso3_cam,
    )
    wide_log = wide.rename(
        columns={
            "alpha_G1": "w_solid",
            "alpha_G2": "w_ww",
            "alpha_G3": "w_res",
        }
    )
    log_waste_family_weights(
        logger, sector="J_Waste", wide=wide_log, focus_iso3=iso3_cam
    )
    pollutants = list(cfg["pollutants"])
    ws, ww, wr = composite_waste.weight_arrays_from_alpha_g123(alpha)

    proxies = build_all_proxies(cfg, ref)
    p_solid = proxies["proxy_solid"]
    p_ww = proxies["proxy_wastewater"]
    p_res = proxies["proxy_residual"]
    _log_raster_stats("proxy_solid", p_solid)
    _log_raster_stats("proxy_wastewater", p_ww)
    _log_raster_stats("proxy_residual", p_res)

    mask = composite_waste.load_mask_optional(cfg, ref)
    comps = composite_waste.composite_per_pollutant(
        p_solid,
        p_ww,
        p_res,
        country_id,
        ws,
        ww,
        wr,
        pollutants,
        mask,
        cfg,
    )

    def _normalize_for_cam_cells(
        cam_cell_id: np.ndarray,
        label: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        weight_bands: dict[str, np.ndarray] = {}
        combined_fb = np.zeros_like(cam_cell_id, dtype=bool)
        fb_summary: list[tuple[str, int]] = []
        for pol in pollutants:
            pl = pol.lower()
            _log_raster_stats(f"composite[{label}][{pl}]", comps[pl])
            wfin, cell_fb = normalize_within_cams_cells(
                comps[pl],
                cam_cell_id,
                context=f"J_Waste pollutant={pl} stage={label}",
                uniform_fallback_summary=fb_summary,
            )
            weight_bands[pl] = wfin
            if cell_fb is not None:
                combined_fb |= cell_fb
            errs = validate_weight_sums(wfin, cam_cell_id)
            if errs[:5]:
                logger.warning("Weight-sum check (%s) sample issues: %s", label, errs[:5])
        if fb_summary:
            tot = sum(c for _, c in fb_summary)
            logger.info(
                "J_Waste (%s): CAMS-cell uniform proxy fallback %d stage(s), %d fine pixels.",
                label,
                len(fb_summary),
                tot,
            )
        return weight_bands, combined_fb

    part = str(cfg.get("waste_build_part", "both")).strip().lower()
    if part not in ("both", "area", "point"):
        part = "both"

    weight_bands_area: dict[str, np.ndarray] = {}
    combined_fb_area = np.zeros_like(cam_cell_id_area, dtype=bool)
    weight_bands_point: dict[str, np.ndarray] = {}

    if part in ("both", "area"):
        weight_bands_area, combined_fb_area = _normalize_for_cam_cells(cam_cell_id_area, "area")
    if part in ("both", "point"):
        weight_bands_point, _ = _normalize_for_cam_cells(cam_cell_id_point, "point")

    area_name, point_name = _output_area_point_names(cfg)
    out_area = out_dir / area_name
    out_point = out_dir / point_name
    if weight_bands_area:
        diagnostics_waste.write_multiband_weights(out_area, weight_bands_area, pollutants, ref)
    if weight_bands_point:
        diagnostics_waste.write_multiband_weights(out_point, weight_bands_point, pollutants, ref)

    diagnostics_waste.write_pollutant_band_mapping(out_dir / "pollutant_band_mapping.csv", pollutants)
    diagnostics_waste.write_country_pollutant_weights(
        out_dir / "country_pollutant_subsector_weights.csv", wide
    )
    if part in ("both", "area") and weight_bands_area:
        diagnostics_waste.write_zero_proxy_diagnostics(
            out_dir / "diagnostics_zero_proxy_cells.csv",
            cam_cell_id_area,
            combined_fb_area,
        )
    diagnostics_waste.write_fallback_log(out_dir / "diagnostics_ceip_fallbacks.csv", None)

    if cfg["output"].get("write_intermediates"):
        diagnostics_waste.write_geotiff_single(out_dir / "proxy_solid.tif", p_solid, ref)
        diagnostics_waste.write_geotiff_single(out_dir / "proxy_wastewater.tif", p_ww, ref)
        diagnostics_waste.write_geotiff_single(out_dir / "proxy_residual.tif", p_res, ref)
        diagnostics_waste.write_geotiff_single(
            out_dir / "diagnostic_imperviousness_valid.tif",
            proxies["imperv_valid_mask"].astype(np.float32),
            ref,
        )
        for pol in pollutants:
            pl = pol.lower()
            diagnostics_waste.write_geotiff_single(
                out_dir / f"composite_proxy_{pl}.tif",
                comps[pl],
                ref,
            )

    logger.info(
        "J_Waste complete (part=%s). Area output: %s | Point output: %s",
        part,
        out_area if weight_bands_area else "(skipped)",
        out_point if weight_bands_point else "(skipped)",
    )
    return {
        "output_tif_area": str(out_area.resolve()),
        "output_tif_point": str(out_point.resolve()),
        "output_dir": str(out_dir.resolve()),
        "waste_build_part": part,
        "wrote_area": bool(weight_bands_area),
        "wrote_point": bool(weight_bands_point),
    }
