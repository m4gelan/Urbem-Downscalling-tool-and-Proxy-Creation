import gc
from dataclasses import replace
from pathlib import Path

import yaml
import numpy as np
from PROXY_V2.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from PROXY_V2.core import log
from PROXY_V2.core.alias import cams_pollutant_var
from PROXY_V2.core.area_weights import build_offroad_S_from_corine, fuse_stationary_offroad_weights
from PROXY_V2.dataset_loaders import require_filepaths_exist
from PROXY_V2.sector.C_Othercombustion.M_matrix import MODEL_CLASSES, build_m_matrix, write_area_weights_debug
from PROXY_V2.sector.C_Othercombustion.U_matrix import assemble_U, log_U_summary, normalize_U_per_cams_cell
from PROXY_V2.sector.C_Othercombustion.X_matrix import build_x_matrix
from PROXY_V2.visualizers.area_weights_map import write_c_othercombustion_area_weights_debug_map
from PROXY_V2.writers.area_weight_stack import write_area_weight_stack_multiband


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
) -> None:
    _ = point_matching

    repo_root = Path(__file__).resolve().parents[3]

    with sector_config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sector config must be a YAML mapping")

    filepaths = cfg.get("filepaths")
    pols = cfg.get("pollutants")
    if not isinstance(pols, list) or not pols:
        raise ValueError("sector config pollutants list missing or empty")

    require_filepaths_exist(repo_root, filepaths, sector_config_path)

    cams_filepath = filepaths.get("CAMS", {}).get("path")
    corine_filepath = filepaths.get("CORINE", {}).get("path")
    population_filepath = filepaths.get("Population", {}).get("path")
    ghs_smod_filepath = filepaths.get("GHS_SMOD", {}).get("path")
    eurostat_end_use_config = filepaths.get("EUROSTAT", {}).get("path")
    eurostat_enabled = bool(cfg.get("eurostat", {}).get("enabled", True))
    gains_folder = filepaths.get("GAINS", {}).get("folder")
    gains_year = filepaths.get("GAINS", {}).get("year")
    gains_mapping_filepath = filepaths.get("GAINS", {}).get("mapping_filepath")
    hotmaps_residential_filepath = filepaths.get("HOTMAPS_RESIDENTIAL", {}).get("path")
    hotmaps_non_residential_filepath = filepaths.get("HOTMAPS_NON_RESIDENTIAL", {}).get("path")
    hdd_filepath = filepaths.get("HDD", {}).get("path")
    emep_yaml_filepath = filepaths.get("EMEP", {}).get("path")

    if point_matching:
        log.info("--------------------------------")
        log.info("No Point Matching for sector C_OtherCombustion")
        log.info("--------------------------------")

    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (C_OtherCombustion)")
        log.info("--------------------------------")

        if not country_profile:
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cfg.get("cams_area_sources")
        year = int(cps_area.get("year", 2019))
        ec = list(cps_area.get("emission_category_indices"))
        st = list(cps_area.get("source_type_indices"))
        pol_list = [str(p).strip() for p in pols if str(p).strip()]

        gains, enduse, m = build_m_matrix(
            repo_root=repo_root,
            gains_folder=gains_folder,
            gains_mapping_filepath=gains_mapping_filepath,
            eurostat_end_use_config=repo_root / str(eurostat_end_use_config).replace("\\", "/"),
            emep_yaml_filepath=emep_yaml_filepath,
            year_gains=gains_year,
            country_profile=country_profile,
            pollutant_outputs=pols,
            eurostat_enabled=eurostat_enabled,
        )
        if log.debug_enabled():
            write_area_weights_debug(
                output_dir / "area_weights_debug.txt",
                gains,
                enduse,
                m,
                pols,
            )

        xb = build_x_matrix(
            repo_root,
            cfg,
            country_profile,
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
            cams_filepath=cams_filepath,
            corine_filepath=corine_filepath,
            population_filepath=population_filepath,
            ghs_smod_filepath=ghs_smod_filepath,
            hotmaps_residential_filepath=hotmaps_residential_filepath,
            hotmaps_non_residential_filepath=hotmaps_non_residential_filepath,
            hdd_filepath=hdd_filepath,
            year=year,
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=pols,
        )
        log.info(f"C_OtherCombustion M matrix shape=({m.shape[0]}, {m.shape[1]})")
        log.info(f"C_OtherCombustion X stack shape=({xb.X.shape[0]}, {xb.X.shape[1]}, {xb.X.shape[2]})")

        U = assemble_U(xb.X, m)
        del m, gains, enduse
        log_U_summary(U, pol_list)
        xb = replace(xb, X=np.empty((0, 0, 0), dtype=np.float32))
        gc.collect()
        W_stationary = normalize_U_per_cams_cell(U, xb.cell_id, xb.cams_cells)
        ch, cw = W_stationary.shape[0], W_stationary.shape[1]
        gc.collect()

        corine_cfg = cfg.get("corine") or {}
        orb = cfg.get("offroad") or {}
        l3_forest = [int(x) for x in corine_cfg.get("l3_forest_agriculture", [])]
        l3_res = [int(x) for x in corine_cfg.get("l3_residential", [])]
        l3_com = [int(x) for x in corine_cfg.get("l3_commercial", [])]
        if not l3_forest or not l3_res or not l3_com:
            raise ValueError("corine l3_forest_agriculture, l3_residential, l3_commercial required for offroad")
        S_forest, S_residential, S_commercial = build_offroad_S_from_corine(
            repo_root,
            corine_filepath,
            l3_forest=l3_forest,
            l3_residential=l3_res,
            l3_commercial=l3_com,
            corine_band=int(corine_cfg.get("band", 1)),
            cams_cells=xb.cams_cells,
            cams_grid=xb.cams_grid,
            ref_height=ch,
            ref_width=cw,
            ref_transform=xb.transform,
            ref_crs=xb.crs,
            pop_z=xb.pop_z,
            residential_w1=float(orb.get("residential_w1")),
            residential_w2=float(orb.get("residential_w2")),
            residential_delta=float(orb.get("residential_delta")),
        )
        log.info(
            f"C_OtherCombustion offroad S sums: forest={float(S_forest.sum()):.6g} "
            f"residential={float(S_residential.sum()):.6g} commercial={float(S_commercial.sum()):.6g}"
        )

        alpha_result = load_sector_alpha_from_config(
            repo_root,
            cfg,
            sector_key="C_OtherCombustion",
            year=year,
            country_profile=country_profile,
            pollutant_labels=pol_list,
        )
        W_stack, W_stat_by_pol, W_off_by_pol, W_comb_by_pol = fuse_stationary_offroad_weights(
            W_stationary,
            S_forest,
            S_residential,
            S_commercial,
            xb.cell_id,
            xb.cams_cells,
            alpha_result,
            alpha_result.pollutant_labels,
        )

        country_tag = country_profile["full_name"].replace(" ", "_")
        out_tif = output_dir / f"C_OtherCombustion_{country_tag}_area_weights_{year}.tif"
        write_area_weight_stack_multiband(
            out_tif,
            W_stack,
            [cams_pollutant_var(p) for p in alpha_result.pollutant_labels],
            xb.transform,
            xb.crs,
        )
        log.info(f"PIPELINE FINISHED: area-weight stack written: {out_tif}")

        if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
            map_html = output_dir / f"C_OtherCombustion_{country_tag}_area_weights_debug_{year}.html"
            try:
                write_c_othercombustion_area_weights_debug_map(
                    map_html,
                    sector_cfg=cfg,
                    pollutant_outputs=pols,
                    model_classes=MODEL_CLASSES,
                    bbox_wgs84=area_weights_viz_bbox_wgs84,
                    x_build=xb,
                    W_stationary_poll=W_stat_by_pol,
                    W_offroad_poll=W_off_by_pol,
                    W_combined_poll=W_comb_by_pol,
                )
                log.info(f"C_OtherCombustion area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"C_OtherCombustion area-weights debug map failed: {exc}")
