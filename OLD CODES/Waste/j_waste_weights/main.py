#!/usr/bin/env python3
"""
CLI entry: build CAMS J_Waste within-cell weight rasters (multi-band GeoTIFF).

Run from repository root::

    python -m Waste.j_waste_weights.main
    python -m Waste.j_waste_weights.main --config Waste/j_waste_weights/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import cams_grid, ceip_weights, composite, country_raster, diagnostics, io_utils, normalization, proxy_building
from .config_loader import load_config, resolve_path


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cams_country_iso3(cfg: dict[str, Any]) -> str:
    cams = cfg.get("cams") or {}
    raw = cams.get("country_iso3")
    if raw:
        return str(raw).strip().upper()
    cntr = (cfg.get("corine_window") or {}).get("nuts_cntr", "EL")
    iso = country_raster.cntr_code_to_iso3(str(cntr))
    return iso or "GRC"


def _output_area_point_names(cfg: dict[str, Any]) -> tuple[str, str]:
    o = cfg.get("output") or {}
    area = str(o.get("weights_tif_area") or o.get("weights_tif") or "Waste_sourcearea.tif")
    point = str(o.get("weights_tif_point") or "Waste_pointarea.tif")
    return area, point


def run_with_config(cfg: dict[str, Any]) -> int:
    """Run pipeline from a loaded config dict (must include ``_project_root``)."""
    log_level = (cfg.get("logging") or {}).get("level", "INFO")
    _setup_logging(str(log_level))

    root: Path = cfg["_project_root"]
    out_dir = resolve_path(root, Path(cfg["output"]["dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = io_utils.load_ref_profile(cfg)
    nc_path = resolve_path(root, Path(cfg["paths"]["cams_nc"]))
    if not nc_path.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc_path}")

    nuts_path = resolve_path(root, Path(cfg["paths"]["nuts_gpkg"]))
    logger = logging.getLogger(__name__)
    logger.info("Reference grid %s x %s, CRS %s", ref["height"], ref["width"], ref["crs"])

    gnfr_j = int((cfg.get("cams") or {}).get("gnfr_j_index", 13))
    iso3_cam = _cams_country_iso3(cfg)
    cam_cell_id_area = cams_grid.build_cam_cell_id_masked_for_j_sources(
        nc_path,
        ref,
        gnfr_j_index=gnfr_j,
        source_type=1,
        country_iso3=iso3_cam,
    )
    cam_cell_id_point = cams_grid.build_cam_cell_id_masked_for_j_sources(
        nc_path,
        ref,
        gnfr_j_index=gnfr_j,
        source_type=2,
        country_iso3=iso3_cam,
    )
    logger.info(
        "CAMS cell masks (GNFR J): area pixels %s, point pixels %s (country %s)",
        int(np.count_nonzero(cam_cell_id_area >= 0)),
        int(np.count_nonzero(cam_cell_id_point >= 0)),
        iso3_cam,
    )

    country_id, iso3_list = country_raster.rasterize_country_ids(nuts_path, ref)

    _, wide, ceip_fb = ceip_weights.build_ceip_weight_tables(cfg)
    pollutants = [str(p) for p in cfg["cams"]["pollutants_nc"]]
    ws, ww, wr = ceip_weights.weights_lookup_arrays(wide, iso3_list, pollutants)

    proxies = proxy_building.build_all_proxies(cfg, ref)
    p_solid = proxies["proxy_solid"]
    p_ww = proxies["proxy_wastewater"]
    p_res = proxies["proxy_residual"]

    mask = composite.load_mask_optional(cfg, ref)
    comps = composite.composite_per_pollutant(
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
        for pol in pollutants:
            pl = pol.lower()
            wfin, cell_fb = normalization.normalize_within_cams_cells(comps[pl], cam_cell_id)
            weight_bands[pl] = wfin
            combined_fb |= cell_fb
            errs = normalization.validate_weight_sums(wfin, cam_cell_id)
            if errs[:5]:
                logger.warning(
                    "Weight-sum check (%s) sample issues: %s", label, errs[:5]
                )
        return weight_bands, combined_fb

    weight_bands_area, combined_fb_area = _normalize_for_cam_cells(cam_cell_id_area, "area")
    weight_bands_point, _ = _normalize_for_cam_cells(cam_cell_id_point, "point")

    area_name, point_name = _output_area_point_names(cfg)
    out_area = out_dir / area_name
    out_point = out_dir / point_name
    diagnostics.write_multiband_weights(out_area, weight_bands_area, pollutants, ref)
    diagnostics.write_multiband_weights(out_point, weight_bands_point, pollutants, ref)

    diagnostics.write_pollutant_band_mapping(out_dir / "pollutant_band_mapping.csv", pollutants)
    diagnostics.write_country_pollutant_weights(out_dir / "country_pollutant_subsector_weights.csv", wide)
    diagnostics.write_zero_proxy_diagnostics(
        out_dir / "diagnostics_zero_proxy_cells.csv",
        cam_cell_id_area,
        combined_fb_area,
    )
    diagnostics.write_fallback_log(out_dir / "diagnostics_ceip_fallbacks.csv", ceip_fb)

    if cfg["output"].get("write_intermediates"):
        diagnostics.write_geotiff_single(out_dir / "proxy_solid.tif", p_solid, ref)
        diagnostics.write_geotiff_single(out_dir / "proxy_wastewater.tif", p_ww, ref)
        diagnostics.write_geotiff_single(out_dir / "proxy_residual.tif", p_res, ref)
        diagnostics.write_geotiff_single(
            out_dir / "diagnostic_imperviousness_valid.tif",
            proxies["imperv_valid_mask"].astype(np.float32),
            ref,
        )
        for pol in pollutants:
            pl = pol.lower()
            diagnostics.write_geotiff_single(
                out_dir / f"composite_proxy_{pl}.tif",
                comps[pl],
                ref,
            )

    logger.info("Done. Area output: %s | Point output: %s", out_area, out_point)
    return 0


def run(cfg_path: str | None) -> int:
    cfg = load_config(cfg_path)
    return run_with_config(cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAMS J_Waste within-cell weight rasters.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: Waste/j_waste_weights/config.yaml).",
    )
    args = p.parse_args(argv)
    try:
        return run(args.config)
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
