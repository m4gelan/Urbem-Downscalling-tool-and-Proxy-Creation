from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.warp import Resampling

from proxy.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from proxy.core import log
from proxy.core.alias import cams_pollutant_var, resolve_osm_filepath
from proxy.core.area_weights import (
    combined_S_waste,
    fuse_alpha_weighted_W_planes,
    normalize_W_per_cams_cell,
)
from proxy.core.point_matching.matching import match_cams_to_facilities_one_to_one, point_match_settings
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_eprtr_points import load_eprtr_points
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_osm import load_osm, rasterize_osm
from proxy.dataset_loaders.load_population import load_population
from proxy.dataset_loaders.load_uwwtd_treatment_plants import (
    load_uwwtd_agglomerations,
    load_uwwtd_treatment_plants,
    load_uwwtd_treatment_plants_raster,
)
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.point_matching.fallback import merge_uwwtd_waste_fallback
from proxy.dataset_loaders.load_waste_rasters import load_ghsl_smod, load_imperviousness
from proxy.visualizers.area_weights_map import (
    write_j_waste_area_weights_debug_map,
    w_pollutant_for_viz,
)
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_stack_multiband
from proxy.writers.point_link import write_cams_facility_link_tif
from proxy.writers.w_groups_export import maybe_export_w_groups
from proxy.core.z_score import z_score_inside


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
    _ = area_weights_viz_bbox_wgs84

    # 1. FIRST STEP: CHECKING EVERYTHING IS OK WITH THE FILEPATHS
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
    eprtr_filepath = filepaths.get("EPRTR", {}).get("path")
    population_filepath = filepaths.get("Population", {}).get("path")
    uwwtd_treatment_plants_filepath = filepaths.get("UWWTD_TreatmentPlants", {}).get("path")
    uwwtd_agglomerations_filepath = filepaths.get("UWWTD_Agglomerations", {}).get("path")
    imperviousness_filepath = filepaths.get("Imperviousness", {}).get("path")
    ghsl_smod_filepath = filepaths.get("GHSL_SMOD", {}).get("path")

    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (J_Waste)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("point_matching needs country_profile from entry")
            raise ValueError("point_matching needs country_profile from entry")

        cps = cfg.get("cams_point_sources") or {}
        year = int(cps.get("year"))
        ec = list(cps.get("emission_category_indices") or [])
        st = list(cps.get("source_type_indices") or [])
        cams_nc = repo_root / str(cams_filepath).replace("\\", "/")
        match_mode, max_match_distance_km, cams_grid_meta = point_match_settings(cps, cams_nc=cams_nc)
        if match_mode == "distance":
            log.info(f"Maximum match distance used is {max_match_distance_km} km")
        else:
            log.info("Point matching mode: same CAMS cell")

        uww_cfg = cfg.get("uwwtd_point_sources") or {}

        log.info(
            "Loading CAMS points:"
            f"\n  File     : {cams_filepath}"
            f"\n  Country  : {country_profile['ISO3']}"
            f"\n  Pollutants: {', '.join(str(x).strip() for x in pols if str(x).strip())}"
        )
        cams_points = load_cams_points(
            cams_nc,
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
        )

        eprtr_cfg = cfg.get("eprtr_point_sources") or {}
        reporting_years = [int(y) for y in eprtr_cfg.get("reportingYears")]
        eprtr_sector_code = int(eprtr_cfg.get("eprtr_sector_code"))
        sub_raw = eprtr_cfg.get("eprtr_sub_sector_codes")
        eprtr_sub_sector_codes = [str(code).strip() for code in sub_raw]

        eprtr_points = load_eprtr_points(
            repo_root / str(eprtr_filepath).replace("\\", "/"),
            reporting_years=reporting_years,
            country_full_name=country_profile["full_name"],
            eprtr_sector_code=eprtr_sector_code,
            eprtr_sub_sector_codes=eprtr_sub_sector_codes,
        )

        if eprtr_points:
            matches = match_cams_to_facilities_one_to_one(
                cams_points,
                eprtr_points,
                match_mode=match_mode,
                max_match_distance_km=max_match_distance_km,
                cams_grid_meta=cams_grid_meta,
                facility_id_field_in_output_rows="eprtr_point_id",
                facility_info_field_in_output_rows="eprtr_point_info",
                log_label_for_facility_dataset="E-PRTR waste",
            )
            for m in matches.values():
                if m.get("matched") == "yes":
                    m.setdefault("match_source", "eprtr")
        else:
            log.info("No E-PRTR waste facilities for this country/filter; CAMS remain unmatched until UWWTD.")
            matches = {
                pid: {
                    "cams": dict(row),
                    "matched": "no",
                    "scoring_value": None,
                    "eprtr_point_id": None,
                    "eprtr_point_info": None,
                }
                for pid, row in cams_points.items()
            }

        n_eprtr = sum(1 for m in matches.values() if m.get("matched") == "yes" and m.get("match_source") == "eprtr")
        unmatched = {pid: dict(matches[pid]["cams"]) for pid in matches if matches[pid].get("matched") != "yes"}

        uwwtd_points: dict[str, dict[str, Any]] = {}
        if unmatched:
            uwwtd_rpt_key = str(country_profile["other"]).strip().upper()
            uwwtd_points = load_uwwtd_treatment_plants(
                repo_root / str(uwwtd_treatment_plants_filepath).replace("\\", "/"),
                rpt_state_key=uwwtd_rpt_key,
                active_only=bool(uww_cfg.get("active_only", True)),
            )
            if uwwtd_points:
                uww_fb = match_cams_to_facilities_one_to_one(
                    unmatched,
                    uwwtd_points,
                    match_mode=match_mode,
                    max_match_distance_km=max_match_distance_km,
                    cams_grid_meta=cams_grid_meta,
                    facility_id_field_in_output_rows="uwwtd_facility_id",
                    facility_info_field_in_output_rows="uwwtd_facility_info",
                    log_label_for_facility_dataset="UWWTD treatment plant",
                )
                merge_uwwtd_waste_fallback(matches, uww_fb)
            else:
                log.info("UWWTD fallback skipped: no treatment plant points for this country.")

        n_uww = sum(1 for m in matches.values() if m.get("match_source") == "uwwtd")
        n_yes = sum(1 for m in matches.values() if m.get("matched") == "yes")
        log.info(
            f"J_Waste point matching: {n_eprtr} CAMS→E-PRTR, {n_uww} via UWWTD, {n_yes}/{len(matches)} total matched."
        )

        country_tag = country_profile["full_name"].replace(" ", "_")
        link_tif = write_cams_facility_link_tif(
            matches,
            output_dir / f"J_Waste_{country_tag}_point_source_{year}.tif",
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        log.info(f"J_Waste point link GeoTIFF: {link_tif}")

        if log.debug_enabled():
            map_html = output_dir / f"J_Waste_{country_tag}_point_match_map.html"
            try:
                write_point_match_map(
                    matches,
                    {},
                    map_html,
                    eprtr_points=eprtr_points,
                    uwwtd_facility_points=uwwtd_points,
                )
                log.info(f"J_Waste point match map (open in browser): {map_html}")
            except Exception as exc:
                log.error(f"J_Waste point match map failed: {exc}")

    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (J_Waste)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("area_weights needs country_profile from entry")
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cfg.get("cams_area_sources")
        year = int(cps_area.get("year", 2019))
        ec = list(cps_area.get("emission_category_indices"))
        st = list(cps_area.get("source_type_indices"))

        corine_cfg = cfg.get("corine") or {}
        sw1 = corine_cfg.get("solid_waste_w1") or {}
        sw2 = corine_cfg.get("solid_waste_w2") or {}
        corine_l3_codes_w1 = [int(x) for x in (sw1.get("l3_codes") or [])]
        corine_l3_codes_w2 = [int(x) for x in (sw2.get("l3_codes") or [])]
        corine_band = int(sw1.get("band") or sw2.get("band") or corine_cfg.get("band", 1))
        if not corine_l3_codes_w1 or not corine_l3_codes_w2:
            raise ValueError("sector config: under 'corine', set non-empty 'l3_codes_w1' and 'l3_codes_w2' (CLC L3 integers)")

        uww_area = cfg.get("uwwtd_area_rasters") or {}
        agg_buf_m = float(uww_area.get("agglomeration_buffer_m", 50.0))
        plant_buf_m = float(uww_area.get("treatment_plant_buffer_m", 500.0))
        imp_band = int(uww_area.get("imperviousness_band", 1))

        ghsl_cfg = cfg.get("ghsl_smod") or {}
        ghsl_band = int(ghsl_cfg.get("band", 1))
        rural_codes = [int(x) for x in (ghsl_cfg.get("rural_codes") or [])]
        if not rural_codes:
            raise ValueError("sector config: ghsl_smod.rural_codes must be a non-empty list of class codes")

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

        if not cams_cells_mask:
            log.warning("J_Waste area_weights: no CAMS area cells for this filter; skipping raster stack.")
        else:
            corine_map_w1, cor_tr, cor_crs, cell_id = load_corine(
                repo_root / str(corine_filepath).replace("\\", "/"),
                corine_l3_codes_w1,
                corine_band,
                cams_cells_mask,
                cams_grid,
            )
            corine_map_w2, _, _, _ = load_corine(
                repo_root / str(corine_filepath).replace("\\", "/"),
                corine_l3_codes_w2,
                corine_band,
                cams_cells_mask,
                cams_grid,
            )
            ch, cw = corine_map_w1.shape

            osm_cfg = cfg["osm"]
            osm_polygons = load_osm(
                repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile),
                cams_cells_mask,
                osm_cfg,
            )
            osm_raster = rasterize_osm(
                osm_polygons,
                ch,
                cw,
                cor_tr,
                cor_crs,
                osm_cfg["rasterize"],
                cams_cells_mask,
            )

            rpt_key = str(country_profile["other"]).strip().upper()
            metric_crs = str(osm_cfg["metric_crs"])
            uwwtd_active = bool((cfg.get("uwwtd_point_sources") or {}).get("active_only", True))

            uwwtd_agg_raster = load_uwwtd_agglomerations(
                repo_root / str(uwwtd_agglomerations_filepath).replace("\\", "/"),
                rpt_state_key=rpt_key,
                metric_crs=metric_crs,
                height=ch,
                width=cw,
                transform=cor_tr,
                raster_crs=cor_crs,
                cams_cells=cams_cells_mask,
                buffer_m=agg_buf_m,
                active_only=uwwtd_active,
                rasterize_cfg=osm_cfg["rasterize"],
            )

            uwwtd_plants_raster = load_uwwtd_treatment_plants_raster(
                repo_root / str(uwwtd_treatment_plants_filepath).replace("\\", "/"),
                rpt_state_key=rpt_key,
                metric_crs=metric_crs,
                height=ch,
                width=cw,
                transform=cor_tr,
                raster_crs=cor_crs,
                cams_cells=cams_cells_mask,
                buffer_m=plant_buf_m,
                active_only=uwwtd_active,
                rasterize_cfg=osm_cfg["rasterize"],
            )

            population_map, _, pop_transform, pop_crs, pop_nodata = load_population(
                repo_root / str(population_filepath).replace("\\", "/"),
                cams_cells_mask,
            )
            population_map = warp_raster_to_grid(
                population_map, pop_transform, pop_crs, ch, cw, cor_tr, cor_crs,
                src_nodata=pop_nodata, dest_init_nan=True, nan_fill=0.0,
            )

            imperviousness, imp_tr, imp_crs, imp_nodata = load_imperviousness(
                repo_root / str(imperviousness_filepath).replace("\\", "/"),
                cams_cells_mask,
                band=imp_band,
            )
            imperviousness = warp_raster_to_grid(
                imperviousness, imp_tr, imp_crs, ch, cw, cor_tr, cor_crs,
                src_nodata=imp_nodata, dest_init_nan=False,
            )
            inside = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells_mask) & np.isfinite(
                imperviousness
            )
            imperviousness_z = z_score_inside(
                imperviousness,
                inside,
                upper_quantile=0.99,
                rescale_to_01=True,
            )
            population_z = z_score_inside(
                population_map,
                inside,
                upper_quantile=0.99,
                rescale_to_01=True,
            )

            rural_mask, ghsl_tr, ghsl_crs = load_ghsl_smod(
                repo_root / str(ghsl_smod_filepath).replace("\\", "/"),
                cams_cells_mask,
                rural_codes=rural_codes,
                band=ghsl_band,
            )
            rural_mask = warp_raster_to_grid(
                rural_mask, ghsl_tr, ghsl_crs, ch, cw, cor_tr, cor_crs,
                resampling=Resampling.nearest, dest_init_nan=False,
            )

            log.info(
                f"J_Waste area-weight rasters on CORINE grid {ch}x{cw}: "
                f"corine_w1 sum={int(np.sum(corine_map_w1))} corine_w2 sum={int(np.sum(corine_map_w2))} "
                f"osm max={float(np.max(osm_raster)):.4f} "
                f"uwwtd_agg max={float(np.max(uwwtd_agg_raster)):.4f} "
                f"uwwtd_plants max={float(np.max(uwwtd_plants_raster)):.4f} "
                f"pop sum={float(np.sum(population_map)):.4g} "
                f"rural_mask sum={float(np.sum(rural_mask)):.4g} "
                f"imperv max={float(np.max(imperviousness)):.4f}"
            )
            aw = cfg.get("weights") or {}
            sw = aw.get("solid_waste") or {}
            ww = aw.get("wastewater") or {}
            rw = aw.get("residual") or {}
            solid_waste_w1 = float(sw["w1"])
            solid_waste_w2 = float(sw["w2"])
            solid_waste_w3 = float(sw["w3"])
            wastewater_w1 = float(ww["w1"])
            wastewater_w2 = float(ww["w2"])
            wastewater_w3 = float(ww["w3"])
            wastewater_w4 = float(ww["w4"])
            residual_w1 = float(rw["w1"])
            residual_w2 = float(rw["w2"])
            residual_w3 = float(rw["w3"])

            S_solid_waste, S_wastewater, S_residual = combined_S_waste(
                corine_map_w1,
                corine_map_w2,
                osm_raster,
                uwwtd_agg_raster,
                uwwtd_plants_raster,
                imperviousness_z,
                rural_mask,
                population_z,
                solid_waste_w1,
                solid_waste_w2,
                solid_waste_w3,
                wastewater_w1,
                wastewater_w2,
                wastewater_w3,
                wastewater_w4,
                residual_w1,
                residual_w2,
                residual_w3,
            )

            W_solid_waste = normalize_W_per_cams_cell(
                S_solid_waste,
                cell_id,
                cams_cells_mask,
            )
            W_wastewater = normalize_W_per_cams_cell(
                S_wastewater,
                cell_id,
                cams_cells_mask,
            )
            W_residual = normalize_W_per_cams_cell(
                S_residual,
                cell_id,
                cams_cells_mask,
            )

            alpha_sector_key = "J_Waste"
            pol_list = [str(x).strip() for x in pols if str(x).strip()]
            alpha_result = load_sector_alpha_from_config(
                repo_root,
                cfg,
                sector_key=alpha_sector_key,
                year=year,
                country_profile=country_profile,
                pollutant_labels=pol_list,
            )
            gi = {name: i for i, name in enumerate(alpha_result.group_names)}
            i_solid = gi["solid_waste"]
            i_ww = gi["wastewater"]
            i_res = gi["residual"]
            log.info(
                f"{alpha_sector_key} alpha matrix shape {alpha_result.alpha.shape} "
                f"(pollutants x groups); methods={alpha_result.methods.tolist()}"
            )
            for j, plab in enumerate(alpha_result.pollutant_labels):
                log.info(
                    f"  {plab}: solid={alpha_result.alpha[j, i_solid]:.4f} "
                    f"ww={alpha_result.alpha[j, i_ww]:.4f} residual={alpha_result.alpha[j, i_res]:.4f} "
                    f"(method {int(alpha_result.methods[j])})"
                )

            a_alpha = alpha_result.alpha.astype(np.float64)
            n_poll = a_alpha.shape[0]
            W_by_branch = {
                "solid_waste": W_solid_waste,
                "wastewater": W_wastewater,
                "residual": W_residual,
            }
            W_per_group = [W_by_branch[g] for g in alpha_result.group_names]
            country_tag = country_profile["full_name"].replace(" ", "_")
            mix_by_group = {
                "solid_waste": {
                    "mixer": "linear3",
                    "weights": {
                        "w1": solid_waste_w1,
                        "w2": solid_waste_w2,
                        "w3": solid_waste_w3,
                    },
                    "terms": {
                        "corine_w1": np.asarray(corine_map_w1, dtype=np.float32),
                        "corine_w2": np.asarray(corine_map_w2, dtype=np.float32),
                        "osm": np.asarray(osm_raster, dtype=np.float32),
                    },
                },
                "wastewater": {
                    "mixer": "linear4",
                    "weights": {
                        "w1": wastewater_w1,
                        "w2": wastewater_w2,
                        "w3": wastewater_w3,
                        "w4": wastewater_w4,
                    },
                    "terms": {
                        "uwwtd_agg": np.asarray(uwwtd_agg_raster, dtype=np.float32),
                        "uwwtd_plants": np.asarray(uwwtd_plants_raster, dtype=np.float32),
                        "population": np.asarray(population_z, dtype=np.float32),
                        "imperviousness": np.asarray(imperviousness_z, dtype=np.float32),
                    },
                },
                "residual": {
                    "mixer": "linear3",
                    "weights": {
                        "w1": residual_w1,
                        "w2": residual_w2,
                        "w3": residual_w3,
                    },
                    "terms": {
                        "population": np.asarray(population_z, dtype=np.float32),
                        "rural": np.asarray(rural_mask, dtype=np.float32),
                        "imperviousness": np.asarray(imperviousness_z, dtype=np.float32),
                    },
                },
            }
            maybe_export_w_groups(
                export_w_groups,
                w_groups_export_root,
                sector_key="J_Waste",
                country_tag=country_tag,
                year=year,
                W_by_group=W_by_branch,
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
            out_w_tif = area_weights_tif_path(output_dir, "J_Waste", country_tag, year)
            write_area_weight_stack_multiband(
                out_w_tif,
                W_poll_stack,
                band_names,
                cor_tr,
                cor_crs,
            )
            log.info(f"J_Waste alpha-fused area weights GeoTIFF: {out_w_tif}")

            if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
                w_poll_map = w_pollutant_for_viz(cfg, W_poll_stack, alpha_result.pollutant_labels)
                map_html = output_dir / f"J_Waste_{country_tag}_area_weights_debug_{year}.html"
                try:
                    write_j_waste_area_weights_debug_map(
                        map_html,
                        bbox_wgs84=area_weights_viz_bbox_wgs84,
                        transform=cor_tr,
                        raster_crs=cor_crs,
                        cams_cells=cams_cells_mask,
                        cell_id=cell_id,
                        osm_raster=osm_raster,
                        corine_l3_132=corine_map_w1.astype(np.uint8),
                        corine_l3_121=corine_map_w2.astype(np.uint8),
                        imperviousness_z=imperviousness_z.astype(np.float64),
                        uwwtd_plants_raster=uwwtd_plants_raster.astype(np.float32),
                        uwwtd_agg_raster=uwwtd_agg_raster.astype(np.float32),
                        rural_mask=rural_mask.astype(np.float32),
                        population_z=population_z.astype(np.float64),
                        W_solid=W_solid_waste.astype(np.float64),
                        W_wastewater=W_wastewater.astype(np.float64),
                        W_residual=W_residual.astype(np.float64),
                        W_pollutant=w_poll_map,
                    )
                    log.info(f"J_Waste area-weights debug map: {map_html}")
                except Exception as exc:
                    log.error(f"J_Waste area-weights debug map failed: {exc}")
