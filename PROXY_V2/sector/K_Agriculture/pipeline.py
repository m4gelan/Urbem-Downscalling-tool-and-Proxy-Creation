from pathlib import Path
import numpy as np
import yaml

from PROXY_V2.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from PROXY_V2.core import log
from PROXY_V2.core.alias import cams_pollutant_var
from PROXY_V2.core.area_weights import normalize_W_per_cams_cell
from PROXY_V2.dataset_loaders import require_filepaths_exist
from PROXY_V2.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask
from PROXY_V2.dataset_loaders.load_corine import load_corine_weighted_l3
from PROXY_V2.writers.debug_dump import KAgAreaWeightsDebug, write_k_agriculture_area_weights_debug
from PROXY_V2.writers.area_weight_stack import write_area_weight_stack_multiband
from PROXY_V2.visualizers.area_weights_map import write_k_agriculture_area_weights_debug_map

from PROXY_V2.sector.K_Agriculture.signals.farm_buildings import build_farm_buildings
from PROXY_V2.sector.K_Agriculture.signals.grazed_pastures import build_grazed_pastures
from PROXY_V2.sector.K_Agriculture.signals.crop_nmvoc import CropNmvocResult, build_crop_nmvoc
from PROXY_V2.sector.K_Agriculture.signals.inorganic_n_fertilizer import (
    InorganicNFertilizerResult, build_inorganic_n_fertilizer,
)
from PROXY_V2.sector.K_Agriculture.signals.livestock_housing_pasture import (
    LivestockHousingPastureResult, build_livestock_housing_pasture,
)
from PROXY_V2.sector.K_Agriculture.signals.biomass_burning import (
    BiomassBurningResult,
    build_biomass_burning,
)
from PROXY_V2.sector.K_Agriculture.signals.manure_application_to_soil import (
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

    if point_matching:
        log.info("NO POINT MATCHING for K_Agriculture")
    if not area_weights:
        return None
    if not country_profile:
        raise ValueError("area_weights needs country_profile from entry")

    log.info("--------------------------------")
    log.info("AREA WEIGHTS (K_Agriculture)")
    log.info("--------------------------------")

    cps_area = cfg.get("cams_area_sources") or {}
    year = int(cps_area.get("year", 2019))
    ec = list(cps_area.get("emission_category_indices") or [])
    st = list(cps_area.get("source_type_indices") or [])

    cams_cells, cams_grid = load_cams_cells_mask(
        repo_root / str(filepaths.get("CAMS", {}).get("path")).replace("\\", "/"),
        year=year,
        country_iso3=country_profile["ISO3"],
        emission_category_indices=ec,
        source_type_indices=st,
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
        osm_filepath=fp.get("OSM", {}).get("path"),
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

    viirs_dir = (fp.get("GFED4") or {}).get("folder")
    if not viirs_dir:
        raise ValueError("filepaths.GFED4.folder required for VIIRS active fire archive")
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

    a_alpha = alpha_result.alpha.astype(np.float32)
    n_poll = a_alpha.shape[0]
    W_poll_stack = np.zeros((n_poll, h, w), dtype=np.float32)
    for j in range(n_poll):
        acc = np.zeros((h, w), dtype=np.float32)
        for i, gname in enumerate(group_names):
            acc += np.float32(a_alpha[j, i]) * W_by_group[gname]
        W_poll_stack[j] = acc

    country_tag = country_profile["full_name"].replace(" ", "_")
    band_names = [cams_pollutant_var(x) for x in alpha_result.pollutant_labels]
    out_w_tif = output_dir / f"K_Agriculture_{country_tag}_area_weights_alpha_{year}.tif"
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

            def _poll_ix(small: str) -> int | None:
                for jj, lab in enumerate(alpha_result.pollutant_labels):
                    if cams_pollutant_var(lab) == small:
                        return jj
                return None

            w_poll: dict[str, np.ndarray] = {}
            for key in ("nh3", "nmvoc"):
                ix = _poll_ix(key)
                if ix is not None:
                    w_poll[key] = W_poll_stack[ix]

            legend_bits: list[str] = []
            for disp, small in (("NH3", "nh3"), ("NMVOC", "nmvoc")):
                ix = _poll_ix(small)
                if ix is None:
                    continue
                row = alpha_result.alpha[ix, :]
                legend_bits.append(
                    "<b>"
                    + disp
                    + "</b> α: "
                    + " ".join(f"{group_names[i]}={float(row[i]):.4f}" for i in range(len(group_names)))
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
                    legend_alpha_html="<br>".join(legend_bits),
                )
                log.info(f"K_Agriculture area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"K_Agriculture area-weights debug map failed: {exc}")

    
