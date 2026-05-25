import gc
from dataclasses import replace
from pathlib import Path

import yaml
import numpy as np
from proxy.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from proxy.core import log
from proxy.core.alias import cams_pollutant_var
from proxy.core.area_weights import (
    build_offroad_S_from_corine,
    fuse_offroad_combined_band,
    log_stationary_alpha,
)
from proxy.dataset_loaders import require_filepaths_exist
from proxy.sector.C_Othercombustion.M_matrix import MODEL_CLASSES, build_m_matrix, write_area_weights_debug
from proxy.sector.C_Othercombustion.U_matrix import stationary_weight_band
from proxy.sector.C_Othercombustion.X_matrix import build_x_matrix
from proxy.visualizers.area_weights_map import (
    parse_c_othercombustion_debug_viz,
    write_c_othercombustion_area_weights_debug_map,
)
from proxy.writers.area_weight_stack import area_weights_tif_path, open_area_weight_stack, write_area_weight_plane


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
        ch, cw = int(xb.X.shape[0]), int(xb.X.shape[1])
        M_rows = np.asarray(m, dtype=np.float32)
        del m, gains, enduse
        gc.collect()

        alpha_result = load_sector_alpha_from_config(
            repo_root,
            cfg,
            sector_key="C_OtherCombustion",
            year=year,
            country_profile=country_profile,
            pollutant_labels=pol_list,
        )
        gix = {g: i for i, g in enumerate(alpha_result.group_names)}
        log_stationary_alpha(alpha_result, pol_list)

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
            alpha_result=alpha_result,
            pollutant_labels=pol_list,
        )

        viz_pollutants: set[str] = set()
        xb_viz = None
        if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
            viz_classes, viz_list = parse_c_othercombustion_debug_viz(cfg, pols, model_classes=MODEL_CLASSES)
            viz_pollutants = set(viz_list)
            xb_viz = replace(
                xb,
                X=np.empty((0, 0, 0), dtype=np.float32),
                stock_by_class={c: xb.stock_by_class[c] for c in viz_classes},
                load_by_class={c: xb.load_by_class[c] for c in viz_classes},
            )
        # U loop only needs X; drop duplicate per-class rasters and z-score layers from RAM. Memory crash otherwise
        xb = replace(
            xb,
            stock_by_class={},
            load_by_class={},
            pop_z=np.empty(0, dtype=np.float32),
            H_res_z=np.empty(0, dtype=np.float32),
            H_nres_z=np.empty(0, dtype=np.float32),
            Hdd_z=np.empty(0, dtype=np.float32),
            u111=np.empty((0, 0), dtype=np.uint8),
            u112=np.empty((0, 0), dtype=np.uint8),
            u121=np.empty((0, 0), dtype=np.uint8),
        )
        gc.collect()

        country_tag = country_profile["full_name"].replace(" ", "_")
        out_tif = area_weights_tif_path(output_dir, "C_Othercombustion", country_tag, year)
        band_labels = [cams_pollutant_var(p) for p in alpha_result.pollutant_labels]
        dst = open_area_weight_stack(out_tif, ch, cw, len(band_labels), xb.transform, xb.crs)

        W_stat_by_pol: dict[str, np.ndarray] = {}
        W_off_by_pol: dict[str, np.ndarray] = {}
        W_comb_by_pol: dict[str, np.ndarray] = {}

        for j, pol in enumerate(alpha_result.pollutant_labels):
            W_s = stationary_weight_band(
                xb.X, M_rows[j], xb.cell_id, xb.cams_cells,
            )
            W_s, W_o, W_c = fuse_offroad_combined_band(
                W_s, S_forest, S_residential, S_commercial,
                xb.cell_id, xb.cams_cells, alpha_result.alpha[j], gix,
            )
            write_area_weight_plane(dst, j + 1, band_labels[j], W_c)
            if pol in viz_pollutants:
                W_stat_by_pol[pol] = W_s
                W_off_by_pol[pol] = W_o
                W_comb_by_pol[pol] = W_c

        dst.close()
        del M_rows, dst, S_forest, S_residential, S_commercial
        xb = replace(xb, X=np.empty((0, 0, 0), dtype=np.float32))
        gc.collect()
        log.info(f"PIPELINE FINISHED: area-weight stack written: {out_tif}")

        if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None and xb_viz is not None:
            map_html = output_dir / f"C_OtherCombustion_{country_tag}_area_weights_debug_{year}.html"
            try:
                write_c_othercombustion_area_weights_debug_map(
                    map_html,
                    sector_cfg=cfg,
                    pollutant_outputs=pols,
                    model_classes=MODEL_CLASSES,
                    bbox_wgs84=area_weights_viz_bbox_wgs84,
                    x_build=xb_viz,
                    W_stationary_poll=W_stat_by_pol,
                    W_offroad_poll=W_off_by_pol,
                    W_combined_poll=W_comb_by_pol,
                )
                log.info(f"C_OtherCombustion area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"C_OtherCombustion area-weights debug map failed: {exc}")
