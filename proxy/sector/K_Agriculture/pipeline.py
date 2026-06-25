from pathlib import Path
import numpy as np
import yaml

from proxy.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from proxy.core import log
from proxy.core.alias import cams_pollutant_var, resolve_osm_filepath
from proxy.core.area_weights import fuse_alpha_weighted_W_planes, normalize_W_per_cams_cell
from proxy.dataset_loaders import require_filepaths_exist
from proxy.core.point_matching.sector_flow import run_sector_point_matching
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.core.cams_sector_config import cams_area_emissions, load_sector_cells_mask
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask
from proxy.writers.point_link import write_cams_facility_link_tif
from proxy.dataset_loaders.load_corine import load_corine_weighted_l3
from proxy.writers.debug_dump import KAgAreaWeightsDebug, write_k_agriculture_area_weights_debug
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_stack_multiband
from proxy.writers.w_groups_export import maybe_export_w_groups
from proxy.visualizers.area_weights_map import (
    alpha_legend_html,
    write_k_agriculture_area_weights_debug_map,
    w_pollutant_for_viz,
)

from proxy.sector.K_Agriculture.signals.farm_buildings import build_farm_buildings
from proxy.sector.K_Agriculture.signals.grazed_pastures import build_grazed_pastures
from proxy.sector.K_Agriculture.signals.crop_nmvoc import CropNmvocResult, build_crop_nmvoc
from proxy.sector.K_Agriculture.signals.inorganic_n_fertilizer import (
    InorganicNFertilizerResult, build_inorganic_n_fertilizer,
)
from proxy.sector.K_Agriculture.signals.livestock_housing_pasture import (
    LivestockHousingPastureResult, build_livestock_housing_pasture,
)
from proxy.sector.K_Agriculture.signals.biomass_burning import (
    BiomassBurningResult,
    build_biomass_burning,
)
from proxy.sector.K_Agriculture.signals.manure_application_to_soil import (
    ManureApplicationResult, build_manure_application_to_soils,
)

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
) -> (
    tuple[
        LivestockHousingPastureResult,
        ManureApplicationResult,
        np.ndarray,
        np.ndarray,
        InorganicNFertilizerResult,
        CropNmvocResult,
        np.ndarray,
        BiomassBurningResult,
    ] | None
):
    repo_root = Path(__file__).resolve().parents[3]
    with sector_config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("sector config must be a YAML mapping")

    filepaths = cfg.get("filepaths")
    pols = cfg.get("pollutants")
    if not isinstance(pols, list) or not pols:
        raise ValueError("sector config pollutants list missing or empty")

    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (K_Agriculture)")
        log.info("--------------------------------")
        if not country_profile:
            raise ValueError("point_matching needs country_profile from entry")
        cps = cfg.get("cams_point_sources") or {}
        year = int(cps.get("year"))
        ec = list(cps.get("emission_category_indices") or [])
        st = list(cps.get("source_type_indices") or [])
        cams_filepath = filepaths.get("CAMS", {}).get("path")
        cams_nc = repo_root / str(cams_filepath).replace("\\", "/")
        pol_labels = [str(x).strip() for x in pols if str(x).strip()]
        cams_points = load_cams_points(
            cams_nc,
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=pol_labels,
        )
        if not cams_points:
            log.info("No CAMS point sources for this sector/country; skipping point matching.")
        else:
            matches = run_sector_point_matching(
                cams_points,
                cfg=cfg,
                repo_root=repo_root,
                country_profile=country_profile,
                pollutant_labels=pol_labels,
                cams_nc=cams_nc,
                fallback_sources=None,
            )
            country_tag = country_profile["full_name"].replace(" ", "_")
            write_cams_facility_link_tif(
                matches,
                output_dir / f"K_Agriculture_{country_tag}_point_source_{year}.tif",
                crs=crs,
                resolution_m=resolution_m,
                pad_m=pad_m,
            )
    if not area_weights:
        return None
    if not country_profile:
        raise ValueError("area_weights needs country_profile from entry")

    require_filepaths_exist(repo_root, filepaths, sector_config_path, country_profile=country_profile)

    log.info("--------------------------------")
    log.info("AREA WEIGHTS (K_Agriculture)")
    log.info("--------------------------------")

    year = int(cams_area_emissions(cfg)["year"])

    cams_cells, cams_grid = load_sector_cells_mask(
        repo_root / str(filepaths.get("CAMS", {}).get("path")).replace("\\", "/"),
        cfg,
        country_iso3=country_profile["ISO3"],
        pollutants=[str(p).strip() for p in pols if str(p).strip()],
        crs=crs,
        resolution_m=resolution_m,
        pad_m=pad_m,
    )

    fp = filepaths
    livestock = build_livestock_housing_pasture(
        repo_root, cfg, country_profile, sector_config_path=sector_config_path,
        cams_cells=cams_cells, cams_grid=cams_grid,
        corine_filepath=fp.get("CORINE", {}).get("path"),
        osm_filepath=resolve_osm_filepath(fp.get("OSM", {}).get("path"), country_profile),
        nuts_filepath=fp.get("NUTS", {}).get("path"),
        farmstock_filepath=fp.get("Farmstock", {}).get("path"),
        lucas_filepath=fp.get("LUCAS", {}).get("path"),
    )
    log.info(
        f"livestock housing/pasture done: grazing shape={livestock.grazing.shape} "
        f"fused max={float(livestock.fused.max()):.6g}"
    )

    manure = build_manure_application_to_soils(
        repo_root, cfg, country_profile, sector_config_path=sector_config_path,
        ref=livestock.ref, cams_cells=cams_cells,
        corine_filepath=fp.get("CORINE", {}).get("path"),
        lucas_filepath=fp.get("LUCAS", {}).get("path"),
        farmstock_filepath=fp.get("Farmstock", {}).get("path"),
    )
    log.info(
        f"manure application done: S sum={float(manure.kg_n_per_pixel_yr.sum()):.6g} "
        f"z max={float(manure.z_scored.max()):.6g}"
    )

    grazed_pastures, _ = build_grazed_pastures(livestock)
    farm_buildings, _ = build_farm_buildings(livestock)

    log.info("--- K_Agriculture signal: broad agricultural ---")
    corine_cfg = cfg.get("corine") or {}
    corine_band = int(corine_cfg["band"])
    broad_w_cfg = (cfg.get("weights") or {}).get("broad_agricultural_signal") or {}
    l3_weights = {
        int(l3): float(broad_w_cfg[f"w_clc{int(l3)}"])
        for l3 in corine_cfg["broad_agricultural_signal"]
    }
    ref = livestock.ref
    broad_raw = load_corine_weighted_l3(
        repo_root / str(fp.get("CORINE", {}).get("path")).replace("\\", "/"),
        l3_weights,
        corine_band,
        cams_cells,
        ref_height=ref.height,
        ref_width=ref.width,
        ref_transform=ref.transform,
        ref_crs=ref.crs,
    )
    mx = float(broad_raw.max())
    broad_agricultural = (
        (broad_raw / np.float32(mx + 1e-12)).astype(np.float32)
        if mx > 0 else broad_raw.astype(np.float32)
    )
    log.info(
        f"broad agricultural: max={float(broad_agricultural.max()):.6g} "
        f"sum={float(broad_agricultural.sum()):.6g}"
    )

    inorganic_n = build_inorganic_n_fertilizer(
        repo_root, cfg, country_profile, sector_config_path=sector_config_path,
        ref=livestock.ref, cams_cells=cams_cells, cams_grid=cams_grid,
        corine_filepath=fp.get("CORINE", {}).get("path"),
        lucas_filepath=fp.get("LUCAS", {}).get("path"),
    )
    log.info(f"inorganic N-fertilizer done: norm max={float(inorganic_n.normalized.max()):.6g}")

    crop_nmvoc = build_crop_nmvoc(
        repo_root, cfg, country_profile, sector_config_path=sector_config_path,
        ref=livestock.ref, cams_cells=cams_cells, cams_grid=cams_grid,
        corine_filepath=fp.get("CORINE", {}).get("path"),
        lucas_filepath=fp.get("LUCAS", {}).get("path"),
    )
    log.info(f"crop NMVOC done: norm max={float(crop_nmvoc.normalized.max()):.6g}")

    viirs_dir = (fp.get("VIIRS") or {}).get("folder")
    if not viirs_dir:
        raise ValueError("filepaths.VIIRS.folder required for VIIRS active fire archive")
    biomass = build_biomass_burning(
        repo_root,
        cfg,
        ref=livestock.ref,
        cams_cells=cams_cells,
        corine_filepath=fp.get("CORINE", {}).get("path"),
        viirs_dir=viirs_dir,
    )
    log.info(f"biomass burning done: z max={float(biomass.z_scored.max()):.6g}")

    ref = livestock.ref
    cell_id = ref.cell_id
    cor_tr = ref.transform
    cor_crs = ref.crs
    h, w = ref.height, ref.width

    S_by_group = {
        "livestock": livestock.fused,
        "manure": manure.z_scored,
        "grazed_pastures": grazed_pastures,
        "farm_buildings": farm_buildings,
        "inorganic_n": inorganic_n.normalized,
        "crop_nmvoc": crop_nmvoc.normalized,
        "broad_agricultural": broad_agricultural,
        "biomass_burning": biomass.z_scored,
    }

    pol_list = [str(x).strip() for x in pols if str(x).strip()]
    alpha_result = load_sector_alpha_from_config(
        repo_root,
        cfg,
        sector_key="K_Agriculture",
        year=year,
        country_profile=country_profile,
        pollutant_labels=pol_list,
    )
    group_names = alpha_result.group_names
    missing = [g for g in group_names if g not in S_by_group]
    if missing:
        raise ValueError(f"alpha groups missing signal rasters: {missing}")

    W_by_group: dict[str, np.ndarray] = {}
    for gname in group_names:
        W_by_group[gname] = normalize_W_per_cams_cell(S_by_group[gname], cell_id, cams_cells)
        log.info(
            f"K_Agriculture W[{gname}]: sum={float(W_by_group[gname].sum()):.6g} "
            f"max={float(W_by_group[gname].max()):.6g}"
        )

    for j, plab in enumerate(alpha_result.pollutant_labels):
        parts = [
            f"{group_names[i]}={float(alpha_result.alpha[j, i]):.4f}"
            for i in range(len(group_names))
        ]
        log.info(f"  {plab}: " + " ".join(parts) + f" (method {int(alpha_result.methods[j])})")

    country_tag = country_profile["full_name"].replace(" ", "_")
    maybe_export_w_groups(
        export_w_groups,
        w_groups_export_root,
        sector_key="K_Agriculture",
        country_tag=country_tag,
        year=year,
        W_by_group=W_by_group,
        cell_id=cell_id,
        transform=cor_tr,
        crs=cor_crs,
        alpha_result=alpha_result,
        cams_cells=cams_cells,
    )

    a_alpha = alpha_result.alpha.astype(np.float32)
    n_poll = a_alpha.shape[0]
    W_per_group = [W_by_group[gname] for gname in group_names]
    W_poll_stack = np.zeros((n_poll, h, w), dtype=np.float32)
    for j in range(n_poll):
        W_poll_stack[j] = fuse_alpha_weighted_W_planes(
            W_per_group, a_alpha[j], cell_id, cams_cells,
        )

    band_names = [cams_pollutant_var(x) for x in alpha_result.pollutant_labels]
    out_w_tif = area_weights_tif_path(output_dir, "K_Agriculture", country_tag, year)
    write_area_weight_stack_multiband(out_w_tif, W_poll_stack, band_names, cor_tr, cor_crs)
    log.info(f"K_Agriculture alpha-fused area weights GeoTIFF: {out_w_tif}")

    if log.debug_enabled():
        dump = KAgAreaWeightsDebug(
            country_iso3=str(country_profile["ISO3"]),
            country_other=str(country_profile["other"]),
            lambda_h=dict(livestock.lambda_by_nuts2),
            manure=list(manure.manure_debug or ()),
            lc1_signals=[
                s
                for s in (inorganic_n.rate_debug, crop_nmvoc.rate_debug)
                if s is not None
            ],
        )
        write_k_agriculture_area_weights_debug(
            output_dir / "area_weights_debug.txt", dump,
        )
        log.info(f"area weights debug written: {output_dir / 'area_weights_debug.txt'}")

        if area_weights_viz_bbox_wgs84 is not None:
            w_poll = w_pollutant_for_viz(cfg, W_poll_stack, alpha_result.pollutant_labels)
            legend_html = alpha_legend_html(
                alpha_result.pollutant_labels,
                alpha_result.alpha,
                group_names,
                cfg,
            )
            map_html = output_dir / f"K_Agriculture_{country_tag}_area_weights_debug_{year}.html"
            try:
                write_k_agriculture_area_weights_debug_map(
                    map_html,
                    bbox_wgs84=area_weights_viz_bbox_wgs84,
                    transform=cor_tr,
                    raster_crs=cor_crs,
                    cams_cells=cams_cells,
                    cell_id=cell_id,
                    W_by_group=W_by_group,
                    W_pollutant=w_poll,
                    legend_alpha_html=legend_html,
                )
                log.info(f"K_Agriculture area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"K_Agriculture area-weights debug map failed: {exc}")

    
