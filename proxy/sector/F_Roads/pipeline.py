from __future__ import annotations

import gc
from pathlib import Path

import yaml

from proxy.core import log
from proxy.core.alias import cams_pollutant_var
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_otm import load_otm_rasters
from proxy.sector.F_Roads.fuel_split import compute_f_road_fuel_split, log_road_fuel_split
from proxy.sector.F_Roads.M_matrix import load_emission_factor_matrices, log_m_by_category, log_pi_fleet
from proxy.sector.F_Roads.weights import build_category_weight_stack, build_x_tot_by_class
from proxy.sector.F_Roads.X_matrix import build_s_tensor, build_x_from_s, model_axes
from proxy.visualizers.area_weights_map import write_f_roads_area_weights_debug_map
from proxy.writers.area_weight_stack import write_area_weight_stack_multiband


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
    _ = (export_w_groups, w_groups_export_root)
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

    require_filepaths_exist(repo_root, filepaths, sector_config_path, country_profile=country_profile)

    if not country_profile:
        raise ValueError("F_Roads needs country_profile from entry")

    road_types, classes, fuels = model_axes(cfg)
    eurostat_path = filepaths.get("EUROSTAT", {}).get("path")
    if not eurostat_path:
        raise KeyError("filepaths.EUROSTAT.path missing in F_Roads sector config")
    eurostat_enabled = bool(cfg.get("eurostat", {}).get("enabled"))

    log.info("--------------------------------")
    log.info("F_Roads Eurostat fuel split")
    log.info("--------------------------------")
    fuel_split = compute_f_road_fuel_split(
        repo_root,
        country_profile,
        repo_root / str(eurostat_path).replace("\\", "/"),
        enabled=eurostat_enabled,
    )
    log_road_fuel_split(fuel_split, country_profile)

    if not area_weights:
        return

    log.info("--------------------------------")
    log.info("F_Roads area weights")
    log.info("--------------------------------")

    cams_filepath = filepaths.get("CAMS", {}).get("path")
    corine_filepath = filepaths.get("CORINE", {}).get("path")
    otm_filepath = filepaths.get("OTM", {}).get("path")
    emep_filepath = filepaths.get("EMEP", {}).get("path")
    cps_area = cfg.get("cams_area_sources")
    f_cats = cfg.get("cams_f_categories")
    year = int(cps_area["year"])
    ec_all = list(cps_area["emission_category_indices"])
    st = list(cps_area["source_type_indices"])
    corine_cfg = cfg.get("corine")
    otm_cfg = cfg.get("otm")
    pol_list = [str(p).strip() for p in pols if str(p).strip()]

    cams_cells, cams_grid = load_cams_cells_mask(
        repo_root / str(cams_filepath).replace("\\", "/"),
        year=year,
        country_iso3=country_profile["ISO3"],
        emission_category_indices=ec_all,
        source_type_indices=st,
        pollutants=pol_list,
        crs=crs,
        resolution_m=resolution_m,
        pad_m=pad_m,
    )
    if not cams_cells:
        raise ValueError("F_Roads: no CAMS cells for country")

    _, cor_tr, cor_crs, cell_id = load_corine(
        repo_root / str(corine_filepath).replace("\\", "/"),
        [int(x) for x in corine_cfg["l3_codes"]],
        int(corine_cfg["band"]),
        cams_cells,
        cams_grid,
    )
    ch, cw = int(cell_id.shape[0]), int(cell_id.shape[1])

    pi, aadt_rasters, roads_by_type, aadt_raw = load_otm_rasters(
        repo_root / str(otm_filepath).replace("\\", "/"),
        country_profile,
        otm_cfg,
        cams_cells,
        height=ch,
        width=cw,
        transform=cor_tr,
        raster_crs=cor_crs,
    )
    log_pi_fleet(pi, road_types, classes)
    s = build_s_tensor(pi, fuel_split, road_types, classes, fuels)
    x = build_x_from_s(s, aadt_rasters, road_types, classes, fuels)
    del s
    gc.collect()
    x_tot = None
    m_exh, m_non = load_emission_factor_matrices(
        repo_root / str(emep_filepath).replace("\\", "/"),
        pol_list,
        classes,
        fuels,
    )
    log_m_by_category(m_exh, m_non, classes, fuels, pol_list, f_cats)

    country_tag = country_profile["full_name"].replace(" ", "_")
    band_names = [cams_pollutant_var(p) for p in pol_list]
    output_dir.mkdir(parents=True, exist_ok=True)

    for cat_name, cat_cfg in f_cats.items():
        cat = str(cat_name)
        ec = [int(cat_cfg["emission_category_index"])]
        cat_cells, _ = load_cams_cells_mask(
            repo_root / str(cams_filepath).replace("\\", "/"),
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=pol_list,
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        cat_fuels = [str(f) for f in cat_cfg["fuels"]] if "fuels" in cat_cfg else None
        if cat == "F4":
            if x_tot is None:
                x_tot = build_x_tot_by_class(pi, aadt_rasters, road_types, classes)
            del x, aadt_rasters
            gc.collect()
        w_stack = build_category_weight_stack(
            x=x if cat != "F4" else None,
            x_tot=x_tot or {},
            m_exh=m_exh,
            m_non=m_non,
            classes=classes,
            fuels=fuels,
            pollutants=pol_list,
            category=cat,
            category_fuels=cat_fuels,
            cell_id=cell_id,
            cams_cells=cat_cells,
        )
        out_tif = output_dir / f"F_Roads_{country_tag}_{cat_name}_{year}.tif"
        write_area_weight_stack_multiband(out_tif, w_stack, band_names, cor_tr, cor_crs)
        log.info(f"F_Roads area weights written: {out_tif}")
        del w_stack, cat_cells
        gc.collect()
        if cat == "F4" and x_tot is not None:
            del x_tot
            x_tot = None
            gc.collect()

    if area_weights_viz_bbox_wgs84 is not None:
        map_stem = output_dir / f"F_Roads_{country_tag}_area_weights_debug_{year}"
        try:
            out_dir = write_f_roads_area_weights_debug_map(
                map_stem,
                bbox_wgs84=area_weights_viz_bbox_wgs84,
                transform=cor_tr,
                raster_crs=cor_crs,
                cams_cells=cams_cells,
                road_types=road_types,
                roads_by_type=roads_by_type,
                aadt_raw=aadt_raw,
            )
            log.info(f"F_Roads viz figures: {out_dir}")
        except Exception as exc:
            log.error(f"F_Roads viz figures failed: {exc}")
