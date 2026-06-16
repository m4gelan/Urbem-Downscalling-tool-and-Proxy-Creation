from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from rasterio.warp import Resampling

from proxy.core import log
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_eprtr_points import load_eprtr_points_energy
from proxy.dataset_loaders.load_jrc_point import load_jrc_points
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_population import load_population
from proxy.core.point_matching.fallback import match_cams_lcp_one_to_one
from proxy.core.point_matching.matching import match_cams_jrc, point_match_settings
from proxy.core.alias import cams_pollutant_var
from proxy.visualizers.area_weights_map import viz_pollutant_labels, write_area_weights_map
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.point_link import write_cams_facility_link_tif
from proxy.core.area_weights import combined_S_publicpower, normalize_W_per_cams_cell
from proxy.core.z_score import z_score_inside
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_equal_multiband
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

    require_filepaths_exist(repo_root, filepaths, sector_config_path)
    
    cams_filepath = filepaths.get("CAMS", {}).get("path")
    # FOR POINT MATCHING
    jrc_filepath = filepaths.get("JRC", {}).get("path")
    lcp_filepath = filepaths.get("LCP", {}).get("path")
    # FOR AREA WEIGHTS
    corine_filepath = filepaths.get("CORINE", {}).get("path")
    population_filepath = filepaths.get("POPULATION", {}).get("path")


    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING")
        log.info("--------------------------------")

        cps = cfg.get("cams_point_sources") 
        if not country_profile:
            log.error("point_matching needs country_profile from entry")
            raise ValueError("point_matching needs country_profile from entry")

        cams_nc = repo_root / cams_filepath.replace("\\", "/")
        match_mode, max_match_distance_km, cams_grid_meta = point_match_settings(cps, cams_nc=cams_nc)
        if match_mode == "distance":
            log.info(f"Maximum match distance used is {max_match_distance_km} km")
        else:
            log.info("Point matching mode: same CAMS cell")

        year = int(cps.get("year"))
        ec = list(cps.get("emission_category_indices") )
        st = list(cps.get("source_type_indices") )

        eps = cfg.get("eprtr_point_sources") 

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

        log.info(
            "Loading JRC points:"
            f"\n  File    : {jrc_filepath}"
        )
        jrc_points = load_jrc_points(
            repo_root / jrc_filepath.replace("\\", "/"),
            year=year,
            country_full_name=country_profile["full_name"],
        )

        log.info("--------------------------------")
        log.info("MATCHING CAMS -> JRC")
        log.info("--------------------------------")
        matches = match_cams_jrc(
            cams_points,
            jrc_points,
            match_mode=match_mode,
            max_match_distance_km=max_match_distance_km,
            cams_grid_meta=cams_grid_meta,
        )

        log.info("--------------------------------")
        log.info("EPRTR/LCP FALLBACK (CAMS not matched to JRC)")
        log.info("--------------------------------")

        log.info(
            "Loading Large Combustion Plants points:"
            f"\n  File    : {lcp_filepath}"
        )
        lcp_points = load_eprtr_points_energy(
            repo_root / lcp_filepath.replace("\\", "/"),
            country_full_name=country_profile["full_name"],
        )

        unmatched_cams = {
            pid: cams_points[pid]
            for pid in cams_points
            if matches.get(pid, {}).get("matched") != "yes"
        }
        lcp_fb = match_cams_lcp_one_to_one(
            unmatched_cams,
            lcp_points,
            match_mode=match_mode,
            max_match_distance_km=max_match_distance_km,
            cams_grid_meta=cams_grid_meta,
        )
        for pid, row in lcp_fb.items():
            matches[pid] = row

        country_tag = country_profile["full_name"].replace(" ", "_")

        point_source_tif = write_cams_facility_link_tif(
            matches,
            output_dir / f"A_PublicPower_{country_tag}_point_source_{year}.tif",
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )

        if log.debug_enabled():
            log.info("Writing point match map...")
            map_html = write_point_match_map(
                matches,
                jrc_points,
                output_dir / f"A_PublicPower_{country_tag}_point_match_map.html",
                eprtr_points=lcp_points,
            )
            log.info(f"Matching map written in : {output_dir}")
    
    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS")
        log.info("--------------------------------")

        if not country_profile:
            log.error("area_weights needs country_profile from entry")
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cfg.get("cams_area_sources")
        year = int(cps_area.get("year", 2019))
        ec = list(cps_area.get("emission_category_indices"))
        st = list(cps_area.get("source_type_indices"))

        corine_cfg = cfg.get("corine") or {}
        corine_l3_codes = [int(x) for x in (corine_cfg.get("l3_codes") or [])]
        corine_band = int(corine_cfg.get("band", 1))
        if not corine_l3_codes:
            raise ValueError("sector config: under 'corine', set non-empty 'l3_codes' (CLC L3 integers)")

        # 1 CAMS area cells + global lon/lat grid metadata (single bundle for downstream)
        cams_cells_mask, cams_grid = load_cams_cells_mask(
            repo_root / cams_filepath.replace("\\", "/"),
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )

        # 2 CORINE on reference grid + per-pixel CAMS cell id (same shape as mask)
        corine_map, cor_tr, cor_crs, cell_id = load_corine(
            repo_root / corine_filepath.replace("\\", "/"),
            corine_l3_codes,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )

        population_map, _, pop_transform, pop_crs, pop_nodata = load_population(
            repo_root / population_filepath.replace("\\", "/"),
            cams_cells_mask,
        )

        # This part ensures that the population raster is on the same grid as the corine raster
        ch, cw = corine_map.shape
        population_map = warp_raster_to_grid(
            population_map, pop_transform, pop_crs, ch, cw, cor_tr, cor_crs,
            src_nodata=pop_nodata, dest_init_nan=True,
        )

        # This part ensures that the population raster is on the same grid as the corine raster
        inside_z = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells_mask) & np.isfinite(
            population_map
        )

        # This part computes the z-score of the population raster
        population_z = z_score_inside(
            population_map,
            inside_z,
            upper_quantile=0.99,
            rescale_to_01=True,
        )

        aw = cfg.get("area_weights") or {}
        w1 = float(aw.get("w1"))
        w2 = float(aw.get("w2"))

        # 3 Combined score S and per-CAMS-cell weights W (cell_id from CORINE loader)
        corine_01 = corine_map.astype(np.float64)
        log.info(f"Raster ready, computing area weights...")
        S = combined_S_publicpower(population_z, corine_01, w1=w1, w2=w2)
        W = normalize_W_per_cams_cell(S, cell_id, cams_cells_mask)

        country_tag = country_profile["full_name"].replace(" ", "_")
        maybe_export_w_groups(
            export_w_groups,
            w_groups_export_root,
            sector_key="A_PublicPower",
            country_tag=country_tag,
            year=year,
            W_by_group={"public_power": W},
            cell_id=cell_id,
            transform=cor_tr,
            crs=cor_crs,
            alpha_result=None,
            cams_cells=cams_cells_mask,
            mix_by_group={
                "public_power": {
                    "mixer": "publicpower",
                    "weights": {"w1": w1, "w2": w2},
                    "terms": {
                        "pop_z": np.asarray(population_z, dtype=np.float32),
                        "corine": np.asarray(corine_01, dtype=np.float32),
                    },
                }
            },
        )

        # 4 Multi-band GeoTIFF on CORINE reference grid
        band_vals = W
        out_tif = area_weights_tif_path(output_dir, "A_PublicPower", country_tag, year)
        write_area_weight_equal_multiband(
            out_tif,
            band_vals,
            [str(x).strip() for x in pols if str(x).strip()],
            cor_tr,
            cor_crs,
        )
        log.info(f"PIPELINE FINISHED: Area-weight stack written: {out_tif}")

        if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
            for pol in viz_pollutant_labels(cfg):
                tag = cams_pollutant_var(pol)
                map_html = output_dir / f"A_PublicPower_{country_tag}_area_weights_debug_{year}_{tag}.html"
                try:
                    write_area_weights_map(
                        map_html,
                        bbox_wgs84=area_weights_viz_bbox_wgs84,
                        corine_map=corine_map,
                        population_z=population_z,
                        W=W,
                        cell_id=cell_id,
                        transform=cor_tr,
                        raster_crs=cor_crs,
                        cams_cells=cams_cells_mask,
                        pollutant_label=pol,
                    )
                    log.info(f"Area-weights debug map: {map_html}")
                except Exception as exc:
                    log.error(f"Area-weights debug map failed ({pol}): {exc}")
