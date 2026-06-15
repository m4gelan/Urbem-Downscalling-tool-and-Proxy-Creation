from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from proxy.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from proxy.core import log
from proxy.core.alias import cams_pollutant_var, resolve_osm_filepath
from proxy.core.area_weights import (
    compute_i_offroad_S_by_subgroup,
    fuse_alpha_weighted_W_planes,
    normalize_W_per_cams_cell,
)
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_osm import load_osm_filtered, rasterize_osm
from proxy.visualizers.area_weights_map import (
    alpha_legend_html,
    write_i_offroad_area_weights_debug_map,
    w_pollutant_for_viz,
)
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_stack_multiband
from proxy.writers.w_groups_export import maybe_export_w_groups


def build(
    output_dir: Path,
    sector_config_path: Path,
    *,
    area_weights: bool = True,
    point_matching: bool = False,
    country_profile: dict[str, str] | None = None,
    crs: str,
    resolution_m: float,
    pad_m: float,
    area_weights_viz_bbox_wgs84: tuple[float, float, float, float] | None = None,
    export_w_groups: bool = False,
    w_groups_export_root: Path | None = None,
) -> None:
    """GNFR I offroad: area weights from OSM + CORINE per subgroup, alpha-fused multi-band GeoTIFF; optional debug map."""

    repo_root = Path(__file__).resolve().parents[3]

    with sector_config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sector config must be a YAML mapping")

    filepaths = cfg.get("filepaths")
    pols = cfg.get("pollutants")
    if not isinstance(pols, list) or not pols:
        raise ValueError("sector config pollutants list missing or empty")

    require_filepaths_exist(repo_root, filepaths, sector_config_path, country_profile=country_profile)

    cams_filepath = filepaths.get("CAMS", {}).get("path")
    corine_filepath = filepaths.get("CORINE", {}).get("path")

    if point_matching:
        log.info("--------------------------------")
        log.info("NO POINT MATCHING NEEDED FOR I_Offroad")
        log.info("--------------------------------")


    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (I_Offroad)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("area_weights needs country_profile from entry")
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cfg.get("cams_area_sources") or {}
        year = int(cps_area.get("year", 2019))
        ec = list(cps_area.get("emission_category_indices") or [])
        st = list(cps_area.get("source_type_indices") or [])

        corine_cfg = cfg.get("corine") or {}
        corine_band = int(corine_cfg.get("band", 1))

        cams_cells_mask, cams_grid = load_cams_cells_mask(
            repo_root / str(cams_filepath).replace("\\", "/"),
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        
        # LOAD the rasters :
        corine_l3_121 = [int(x) for x in (corine_cfg.get("l3_codes_121") or [])]
        corine_l3_123 = [int(x) for x in (corine_cfg.get("l3_codes_123") or [])]
        corine_l3_124 = [int(x) for x in (corine_cfg.get("l3_codes_124") or [])]
        corine_l3_131 = [int(x) for x in (corine_cfg.get("l3_codes_131") or [])]
        
        corine_map_121, cor_tr, cor_crs, cell_id = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_121,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_123, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_123,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_124, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_124,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_131, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_131,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )

        ch, cw = corine_map_121.shape

        osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]] | None = None

        if not cams_cells_mask:
            log.warning("I_Offroad area_weights: no CAMS cells; skipping OSM rasters.")
        else:
            osm_cfg = cfg.get("osm") or {}
            sub_osm = osm_cfg.get("subgroups") or {}
            osm_gpkg = repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile)
            rz = osm_cfg.get("rasterize") or {}
            osm_rasters_by_subgroup = {}

            for sg_name, sg_body in sub_osm.items():
                slots = (sg_body or {}).get("slots") or []
                if not isinstance(slots, list):
                    raise TypeError(f"osm.subgroups.{sg_name}.slots must be a list")
                osm_rasters_by_subgroup[sg_name] = {}
                for slot in slots:
                    sid = str(slot.get("id", "")).strip() or "slot"
                    match = slot.get("match") or {}
                    buf_ov = slot.get("buffer_m") if isinstance(slot.get("buffer_m"), dict) else None
                    gdf = load_osm_filtered(
                        osm_gpkg,
                        cams_cells_mask,
                        osm_cfg,
                        match=match,
                        buffer_m_override=buf_ov,
                    )
                    osm_r = rasterize_osm(
                        gdf,
                        ch,
                        cw,
                        cor_tr,
                        cor_crs,
                        rz,
                        cams_cells_mask,
                    )
                    osm_rasters_by_subgroup[sg_name][sid] = np.asarray(
                        osm_r, dtype=np.float32, order="C", copy=True
                    )
                    log.info(
                        f"I_Offroad OSM subgroup={sg_name} slot={sid} features={len(gdf)} grid_sum={float(np.sum(osm_r)):.6g}"
                    )


            weights_cfg = cfg.get("weights") or {}
            S_by_subgroup = compute_i_offroad_S_by_subgroup(
                osm_rasters_by_subgroup=osm_rasters_by_subgroup,
                corine_121=corine_map_121,
                corine_123=corine_map_123,
                corine_131=corine_map_131,
                corine_124=corine_map_124,
                weights=weights_cfg,
            )
            W_by_subgroup: dict[str, np.ndarray] = {}
            for gname, S_g in S_by_subgroup.items():
                W_g = normalize_W_per_cams_cell(S_g, cell_id, cams_cells_mask)
                W_by_subgroup[gname] = W_g
                log.info(
                    "I_Offroad subgroup=%s S_sum=%.6g W_sum=%.6g",
                    gname,
                    float(np.sum(S_g)),
                    float(np.sum(W_g)),
                )

            alpha_sector_key = "I_Offroad"
            pol_list = [str(x).strip() for x in pols if str(x).strip()]
            alpha_result = load_sector_alpha_from_config(
                repo_root,
                cfg,
                sector_key=alpha_sector_key,
                year=year,
                country_profile=country_profile,
                pollutant_labels=pol_list,
            )
            group_names = alpha_result.group_names
            log.info(
                f"{alpha_sector_key} alpha matrix shape {alpha_result.alpha.shape} "
                f"(pollutants x groups); methods={alpha_result.methods.tolist()}"
            )
            for j, plab in enumerate(alpha_result.pollutant_labels):
                parts = [
                    f"{alpha_result.group_names[i]}={float(alpha_result.alpha[j, i]):.4f}"
                    for i in range(len(alpha_result.group_names))
                ]
                log.info(f"  {plab}: " + " ".join(parts) + f" (method {int(alpha_result.methods[j])})")

            for g in group_names:
                if g not in W_by_subgroup:
                    raise ValueError(
                        f"alpha group {g!r} has no W raster; keys W_by_subgroup={sorted(W_by_subgroup)}"
                    )
            W_per_group = [W_by_subgroup[g] for g in group_names]

            a_alpha = alpha_result.alpha.astype(np.float64)
            n_poll = a_alpha.shape[0]
            n_g = len(W_per_group)
            if a_alpha.shape[1] != n_g:
                raise ValueError(f"alpha columns {a_alpha.shape[1]} != spatial groups {n_g}")

            country_tag = country_profile["full_name"].replace(" ", "_")
            wr = weights_cfg.get("rail_transport") or {}
            wp = weights_cfg.get("pipeline_transport") or {}
            wm = weights_cfg.get("non_road_machinery") or {}
            osg = osm_rasters_by_subgroup
            mix_by_group = {
                "rail_transport": {
                    "mixer": "linear",
                    "weight_keys": ["w_osm_diesel", "w_osm_electric", "w_osm_yard"],
                    "weights": {k: float(wr[k]) for k in ("w_osm_diesel", "w_osm_electric", "w_osm_yard")},
                    "terms": {
                        "rail_diesel": np.asarray(osg["rail_transport"]["rail_diesel"], dtype=np.float32),
                        "rail_electric": np.asarray(osg["rail_transport"]["rail_electric"], dtype=np.float32),
                        "rail_yards": np.asarray(osg["rail_transport"]["rail_yards"], dtype=np.float32),
                    },
                },
                "pipeline_transport": {
                    "mixer": "linear",
                    "weight_keys": ["w_osm_oil_gas_facilities", "w_osm_pipeline_hydrocarbon"],
                    "weights": {k: float(wp[k]) for k in ("w_osm_oil_gas_facilities", "w_osm_pipeline_hydrocarbon")},
                    "terms": {
                        "oil_gas_facilities": np.asarray(osg["pipeline_transport"]["oil_gas_facilities"], dtype=np.float32),
                        "pipeline_hydrocarbon": np.asarray(osg["pipeline_transport"]["pipeline_hydrocarbon"], dtype=np.float32),
                    },
                },
                "non_road_machinery": {
                    "mixer": "linear",
                    "weight_keys": ["w_clc_121", "w_clc_123", "w_clc_124", "w_clc_131"],
                    "weights": {k: float(wm[k]) for k in ("w_clc_121", "w_clc_123", "w_clc_124", "w_clc_131")},
                    "terms": {
                        "clc_121": np.asarray(corine_map_121, dtype=np.float32),
                        "clc_123": np.asarray(corine_map_123, dtype=np.float32),
                        "clc_124": np.asarray(corine_map_124, dtype=np.float32),
                        "clc_131": np.asarray(corine_map_131, dtype=np.float32),
                    },
                },
            }
            maybe_export_w_groups(
                export_w_groups,
                w_groups_export_root,
                sector_key="I_Offroad",
                country_tag=country_tag,
                year=year,
                W_by_group=W_by_subgroup,
                cell_id=cell_id,
                transform=cor_tr,
                crs=cor_crs,
                alpha_result=alpha_result,
                cams_cells=cams_cells_mask,
                mix_by_group=mix_by_group,
            )

            W_poll_stack = np.zeros((n_poll, ch, cw), dtype=np.float32)
            for j in range(n_poll):
                W_poll_stack[j] = fuse_alpha_weighted_W_planes(
                    W_per_group, a_alpha[j], cell_id, cams_cells_mask,
                )

            band_names = [cams_pollutant_var(x) for x in alpha_result.pollutant_labels]
            out_w_tif = area_weights_tif_path(output_dir, "I_Offroad", country_tag, year)
            write_area_weight_stack_multiband(
                out_w_tif,
                W_poll_stack,
                band_names,
                cor_tr,
                cor_crs,
            )
            log.info(f"I_Offroad alpha-fused area weights GeoTIFF: {out_w_tif}")

            if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
                w_poll = w_pollutant_for_viz(cfg, W_poll_stack, alpha_result.pollutant_labels)
                legend_alpha_html = alpha_legend_html(
                    alpha_result.pollutant_labels,
                    alpha_result.alpha,
                    group_names,
                    cfg,
                )
                map_html = output_dir / f"I_Offroad_{country_tag}_area_weights_debug_{year}.html"
                try:
                    write_i_offroad_area_weights_debug_map(
                        map_html,
                        bbox_wgs84=area_weights_viz_bbox_wgs84,
                        transform=cor_tr,
                        raster_crs=cor_crs,
                        cams_cells=cams_cells_mask,
                        cell_id=cell_id,
                        corine_l3_121=corine_map_121,
                        corine_l3_123=corine_map_123,
                        corine_l3_124=corine_map_124,
                        corine_l3_131=corine_map_131,
                        osm_by_subgroup=osm_rasters_by_subgroup,
                        W_by_subgroup=W_by_subgroup,
                        W_pollutant=w_poll,
                        legend_alpha_html=legend_alpha_html,
                    )
                    log.info(f"I_Offroad area-weights debug map: {map_html}")
                except Exception as exc:
                    log.error(f"I_Offroad area-weights debug map failed: {exc}")