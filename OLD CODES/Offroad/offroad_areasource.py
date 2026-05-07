"""
GNFR I (off-road) fine-grid area weights: rail + pipeline + non-road proxies combined with
CEIP/NIR national shares (1A3c / 1A3ei / 1A3eii), then normalized within each CAMS grid cell.

Implementation is split across ``rail``, ``pipeline``, ``nonroad``, ``ceip``, and helpers in
``common``, ``corine_masks``, ``country_grid``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from Shipping.shipping_areasource import resolve_corine_tif, warp_corine_codes_nearest
from Waste.j_waste_weights.cams_grid import build_cam_cell_id
from Waste.j_waste_weights.normalization import normalize_within_cams_cells

from Offroad.ceip import build_share_arrays, read_ceip_shares
from Offroad.constants import IDX_GNFR_I
from Offroad.country_grid import rasterize_country_indices
from Offroad.nonroad import build_nonroad_proxy, warp_population_to_ref
from Offroad.pipeline import build_pipeline_raw_z, build_pipeline_z_final
from Offroad.rail import build_rail_coverage_and_z

logger = logging.getLogger(__name__)

__all__ = ["IDX_GNFR_I", "diagnostics_for_extent", "run_offroad_areasource"]


def run_offroad_areasource(
    *,
    root: Path,
    ref: dict[str, Any],
    yaml_cfg: dict[str, Any],
    paths: dict[str, Any],
    output_tif: Path,
    pollutants: list[str] | None = None,
) -> Path:
    """Build multi-band ``Offroad_Sourcearea.tif``; band order = ``pollutants``."""
    pcfg = yaml_cfg.get("proxy") or {}
    cntr_map = dict(yaml_cfg.get("cntr_code_to_iso3") or {})
    defs = yaml_cfg.get("defaults") or {}
    fallback_iso = str(defs.get("fallback_country_iso3", "GRC")).upper()
    default_triple = (
        float(defs.get("default_shares_rail", 1.0 / 3)),
        float(defs.get("default_shares_pipe", 1.0 / 3)),
        float(defs.get("default_shares_nonroad", 1.0 / 3)),
    )

    pols = pollutants or [str(x) for x in (yaml_cfg.get("pollutants") or ["nox", "pm2_5"])]
    from Offroad.ceip import _norm_pol as norm_pol

    pols_norm = [norm_pol(p) for p in pols]

    aliases = dict(yaml_cfg.get("ceip_pollutant_aliases") or {})
    for k, v in list(aliases.items()):
        aliases[str(k).upper()] = norm_pol(v)

    cor_p = Path(paths["corine"])
    if not cor_p.is_absolute():
        cor_p = root / cor_p
    corine_path = resolve_corine_tif(cor_p)
    osm_gpkg = Path(paths["osm_gpkg"])
    if not osm_gpkg.is_absolute():
        osm_gpkg = root / osm_gpkg
    nuts_gpkg = Path(paths["nuts_gpkg"])
    if not nuts_gpkg.is_absolute():
        nuts_gpkg = root / nuts_gpkg
    pop_path = Path(paths.get("population_tif") or paths.get("pop_path", ""))
    ceip_path = Path(paths["ceip_xlsx"])
    if not ceip_path.is_absolute():
        ceip_path = root / ceip_path
    cams_nc = Path(paths["cams_nc"])
    if not cams_nc.is_absolute():
        cams_nc = root / cams_nc

    ceip_sheet = paths.get("ceip_sheet")
    if isinstance(ceip_sheet, str) and ceip_sheet.lower() in ("null", "none"):
        ceip_sheet = None

    rail_buf = float(pcfg.get("rail_buffer_m", 150))
    subdiv = int(pcfg.get("osm_subdivide", 4))

    clc_nn = warp_corine_codes_nearest(corine_path, ref)

    _, z_rail = build_rail_coverage_and_z(
        osm_gpkg,
        ref,
        rail_buffer_m=rail_buf,
        osm_subdivide=subdiv,
    )

    z_pipe, _, _ = build_pipeline_z_final(osm_gpkg, root, ref, paths, pcfg)

    pop_raw = warp_population_to_ref(pop_path, root, ref)
    nr = build_nonroad_proxy(
        clc_nn=clc_nn,
        ref=ref,
        osm_gpkg=osm_gpkg,
        proxy_cfg=pcfg,
        pop_raw=pop_raw,
    )
    p_nr = nr["p_nr"]

    ceip_year_raw = defs.get("ceip_year")
    ceip_year: int | None
    if ceip_year_raw is None or str(ceip_year_raw).strip() == "":
        ceip_year = None
    else:
        ceip_year = int(ceip_year_raw)

    share_dict = read_ceip_shares(
        ceip_path,
        sheet=ceip_sheet if isinstance(ceip_sheet, str) else None,
        pollutant_aliases=aliases,
        pollutants_wanted=pols_norm,
        cntr_code_to_iso3=cntr_map,
        default_triple=default_triple,
        ceip_year=ceip_year,
    )

    country_idx, idx_to_iso = rasterize_country_indices(nuts_gpkg, ref, cntr_map, fallback_iso)

    cam_id = build_cam_cell_id(cams_nc, ref)

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(ref["height"]),
        "width": int(ref["width"]),
        "count": len(pols_norm),
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": ref["transform"],
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(output_tif, "w", **profile) as dst:
        for bi, pol in enumerate(pols_norm, start=1):
            sr, sp, sn = build_share_arrays(
                country_idx,
                idx_to_iso,
                share_dict,
                pol,
                fallback_iso,
                default_triple=default_triple,
            )
            P = (sr * z_rail + sp * z_pipe + sn * p_nr).astype(np.float32)
            W, _fb = normalize_within_cams_cells(P, cam_id, None)
            dst.write(np.asarray(W, dtype=np.float32), bi)
            dst.set_band_description(bi, pol)

    logger.info("Wrote %s (%d bands)", output_tif, len(pols_norm))
    return output_tif


def diagnostics_for_extent(
    root: Path,
    yaml_cfg: dict[str, Any],
    paths: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Intermediate rasters on ``ref`` (for visualization / QA)."""
    pcfg = yaml_cfg.get("proxy") or {}
    osm_gpkg = Path(paths["osm_gpkg"])
    if not osm_gpkg.is_absolute():
        osm_gpkg = root / osm_gpkg

    corine_path = resolve_corine_tif(
        Path(paths["corine"]) if Path(paths["corine"]).is_file() else root / paths["corine"]
    )
    rail_buf = float(pcfg.get("rail_buffer_m", 150))
    pipe_buf = float(pcfg.get("pipeline_buffer_m", 75))
    subdiv = int(pcfg.get("osm_subdivide", 4))

    rail_cov, z_rail = build_rail_coverage_and_z(
        osm_gpkg,
        ref,
        rail_buffer_m=rail_buf,
        osm_subdivide=subdiv,
    )

    pipe_cov, z_pipe = build_pipeline_raw_z(
        osm_gpkg,
        root,
        ref,
        pipeline_buffer_m=pipe_buf,
        osm_subdivide=subdiv,
    )

    clc_nn = warp_corine_codes_nearest(corine_path, ref)
    pop_path = Path(paths.get("population_tif") or "")
    pop_raw = warp_population_to_ref(pop_path, root, ref)
    nr = build_nonroad_proxy(
        clc_nn=clc_nn,
        ref=ref,
        osm_gpkg=osm_gpkg,
        proxy_cfg=pcfg,
        pop_raw=pop_raw,
    )

    return {
        "z_rail": z_rail,
        "z_pipeline": z_pipe,
        "z_agri": nr["z_agri"],
        "z_industry_combined": nr["z_industry_combined"],
        "z_pop_nonroad": nr["z_pop_nonroad"],
        "p_nonroad_raw": nr["p_nr"],
        "rail_coverage_raw": rail_cov,
        "pipeline_coverage_raw": pipe_cov,
    }
