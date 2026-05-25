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
    compute_d_fugitive_S_by_subgroup,
    fuse_alpha_weighted_W_planes,
    normalize_W_per_cams_cell,
)
from proxy.core.point_matching.matching import match_cams_to_facilities_one_to_one
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_eprtr_points import load_eprtr_points
from proxy.dataset_loaders.load_gem import load_coal_mine_tracker_mask, load_oil_gas_extractors_mask
from proxy.dataset_loaders.load_osm import load_osm_filtered, rasterize_osm
from proxy.dataset_loaders.load_population import load_population
from proxy.dataset_loaders.load_viirs import load_vnf_nightfire_buffer_mask
from proxy.visualizers.area_weights_map import write_d_fugitive_area_weights_debug_map
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_stack_multiband
from proxy.writers.point_link import write_cams_facility_link_tif


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
    """GNFR D fugitive: optional CAMS↔E-PRTR point links; area weights = OSM+CORINE+pop per group, alpha-fused GeoTIFF."""

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
    gol_coal_mine_tracker_filepath = filepaths.get("GLOBAL_COAL_MINE_TRACKER", {}).get("path")
    gol_oil_gas_extractors_filepath = filepaths.get("GLOBAL_OIL_GAS_EXTRACTORS", {}).get("path")
    vnf_filepath = filepaths.get("VNF", {}).get("path")

    country_full_name = country_profile["full_name"]
    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (D_Fugitive)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("point_matching needs country_profile from entry")
            raise ValueError("point_matching needs country_profile from entry")

        cps = cfg.get("cams_point_sources") or {}
        year = int(cps.get("year"))
        ec = list(cps.get("emission_category_indices") or [])
        st = list(cps.get("source_type_indices") or [])
        max_match_distance_km = float(cps.get("max_match_distance_km", 10.0))

        log.info(
            "Loading CAMS points:"
            f"\n  File     : {cams_filepath}"
            f"\n  Country  : {country_profile['ISO3']}"
            f"\n  Pollutants: {', '.join(str(x).strip() for x in pols if str(x).strip())}"
        )
        cams_points = load_cams_points(
            repo_root / str(cams_filepath).replace("\\", "/"),
            year=year,
            country_iso3=country_profile["ISO3"],
            emission_category_indices=ec,
            source_type_indices=st,
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
        )

        eprtr_cfg = cfg.get("eprtr_point_sources") or {}
        reporting_years = [int(y) for y in eprtr_cfg.get("reportingYears")]
        eprtr_sector_raw = eprtr_cfg.get("eprtr_sector_code")
        if eprtr_sector_raw is None:
            raise ValueError("eprtr_point_sources.eprtr_sector_code is required for point matching")
        sub_raw = eprtr_cfg.get("eprtr_sub_sector_codes")
        eprtr_sub_sector_codes = [str(code).strip() for code in sub_raw]

        eprtr_points = load_eprtr_points(
            repo_root / str(eprtr_filepath).replace("\\", "/"),
            reporting_years=reporting_years,
            country_full_name=country_profile["full_name"],
            eprtr_sector_code=eprtr_sector_raw,
            eprtr_sub_sector_codes=eprtr_sub_sector_codes,
        )
        log.info(f"Loaded {len(eprtr_points)} E-PRTR industrial facilities")
        if eprtr_points:
            matches = match_cams_to_facilities_one_to_one(
                cams_points,
                eprtr_points,
                max_match_distance_km=max_match_distance_km,
                facility_id_field_in_output_rows="eprtr_point_id",
                facility_info_field_in_output_rows="eprtr_point_info",
                log_label_for_facility_dataset="E-PRTR industry",
            )
            for m in matches.values():
                if m.get("matched") == "yes":
                    m.setdefault("match_source", "eprtr")
        else:
            log.info("No E-PRTR industrial facilities for this country/filter; CAMS remain unmatched.")
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
        n_yes = sum(1 for m in matches.values() if m.get("matched") == "yes")
        log.info(f"D_Fugitive point matching: {n_eprtr} CAMS→E-PRTR, {n_yes}/{len(matches)} matched.")

        country_tag = country_profile["full_name"].replace(" ", "_")
        link_tif = write_cams_facility_link_tif(
            matches,
            output_dir / f"D_Fugitive_{country_tag}_point_source_{year}.tif",
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        log.info(f"D_Fugitive point link GeoTIFF: {link_tif}")

        if log.debug_enabled():
            map_html = output_dir / f"D_Fugitive_{country_tag}_point_match_map.html"
            try:
                write_point_match_map(
                    matches,
                    {},
                    map_html,
                    eprtr_points=eprtr_points,
                )
                log.info(f"D_Fugitive point match map (open in browser): {map_html}")
            except Exception as exc:
                log.error(f"D_Fugitive point match map failed: {exc}")

    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (D_Fugitive)")
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

        population_filepath = filepaths.get("Population", {}).get("path")
        
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
        corine_l3_131 = [int(x) for x in (corine_cfg.get("l3_codes_131") or [])]
        corine_map_121, cor_tr, cor_crs, cell_id = load_corine(
            repo_root / corine_filepath.replace("\\", "/"),
            corine_l3_121,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_123, _, _, _ = load_corine(
            repo_root / corine_filepath.replace("\\", "/"),
            corine_l3_123,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_131, _, _, _ = load_corine(
            repo_root / corine_filepath.replace("\\", "/"),
            corine_l3_131,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )        
        population_map, _, pop_transform, pop_crs, pop_nodata = load_population(
            repo_root / population_filepath.replace("\\", "/"),
            cams_cells_mask,
        )

        ch, cw = corine_map_121.shape

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

        osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]] | None = None

        if not cams_cells_mask:
            log.warning("D_Fugitive area_weights: no CAMS cells; skipping OSM/GEM/VNF rasters.")
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
                        f"D_Fugitive OSM subgroup={sg_name} slot={sid} features={len(gdf)} grid_sum={float(np.sum(osm_r)):.6g}"
                    )

            gem_coal_cfg = cfg.get("gem_coal") or {}
            gem_og_cfg = cfg.get("gem_oil_gas") or {}
            vnf_cfg = cfg.get("vnf") or {}

            coal_m = load_coal_mine_tracker_mask(
                repo_root / str(gol_coal_mine_tracker_filepath).replace("\\", "/"),
                country_profile["full_name"],
                ch,
                cw,
                cor_tr,
                cor_crs,
                cams_cells_mask,
                gem_coal_cfg,
            )
            log.info(f"D_Fugitive GEM coal binary grid_sum={float(np.sum(coal_m)):.6g}")

            og_m = load_oil_gas_extractors_mask(
                repo_root / str(gol_oil_gas_extractors_filepath).replace("\\", "/"),
                country_profile["full_name"],
                ch,
                cw,
                cor_tr,
                cor_crs,
                cams_cells_mask,
                gem_og_cfg,
            )
            log.info(f"D_Fugitive GEM oil/gas binary grid_sum={float(np.sum(og_m)):.6g}")

            vnf_m = load_vnf_nightfire_buffer_mask(
                repo_root / str(vnf_filepath).replace("\\", "/"),
                ch,
                cw,
                cor_tr,
                cor_crs,
                cams_cells_mask,
                vnf_cfg,
            )
            log.info(f"D_Fugitive VNF buffer binary grid_sum={float(np.sum(vnf_m)):.6g}")
            osm_slot_keys = {k: list(v.keys()) for k, v in osm_rasters_by_subgroup.items()}
            log.info(f"D_Fugitive OSM raster dict keys: {osm_slot_keys}")

            weights_cfg = cfg.get("weights") or {}
            S_by_subgroup = compute_d_fugitive_S_by_subgroup(
                osm_rasters_by_subgroup=osm_rasters_by_subgroup,
                corine_121=corine_map_121,
                corine_123=corine_map_123,
                corine_131=corine_map_131,
                population_z=population_z,
                gem_coal=coal_m,
                gem_oil=og_m,
                vnf=vnf_m,
                weights=weights_cfg,
            )
            W_by_subgroup: dict[str, np.ndarray] = {}
            for gname, S_g in S_by_subgroup.items():
                W_g = normalize_W_per_cams_cell(S_g, cell_id, cams_cells_mask)
                W_by_subgroup[gname] = W_g
                log.info(
                    "D_Fugitive subgroup=%s S_sum=%.6g W_sum=%.6g",
                    gname,
                    float(np.sum(S_g)),
                    float(np.sum(W_g)),
                )

            alpha_sector_key = "D_Fugitive"
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

            W_poll_stack = np.zeros((n_poll, ch, cw), dtype=np.float32)
            for j in range(n_poll):
                W_poll_stack[j] = fuse_alpha_weighted_W_planes(
                    W_per_group, a_alpha[j], cell_id, cams_cells_mask,
                )

            country_tag = country_profile["full_name"].replace(" ", "_")
            band_names = [cams_pollutant_var(x) for x in alpha_result.pollutant_labels]
            out_w_tif = area_weights_tif_path(output_dir, "D_Fugitive", country_tag, year)
            write_area_weight_stack_multiband(
                out_w_tif,
                W_poll_stack,
                band_names,
                cor_tr,
                cor_crs,
            )
            log.info(f"D_Fugitive alpha-fused area weights GeoTIFF: {out_w_tif}")

            if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:

                def _pollutant_row_index(want_small: str) -> int | None:
                    for jj, lab in enumerate(alpha_result.pollutant_labels):
                        if cams_pollutant_var(lab) == want_small:
                            return jj
                    return None

                w_poll: dict[str, np.ndarray] = {}
                for key in ("nmvoc", "nox"):
                    ix = _pollutant_row_index(key)
                    if ix is not None:
                        w_poll[key] = W_poll_stack[ix]

                legend_bits: list[str] = []
                for disp, small in (("NMVOC", "nmvoc"), ("NOx", "nox")):
                    ix = _pollutant_row_index(small)
                    if ix is None:
                        continue
                    row = alpha_result.alpha[ix, :]
                    legend_bits.append(
                        "<b>"
                        + disp
                        + "</b> &alpha;: "
                        + " ".join(f"{group_names[i]}={float(row[i]):.4f}" for i in range(n_g))
                    )
                legend_alpha_html = "<br>".join(legend_bits) if legend_bits else ""

                map_html = output_dir / f"D_Fugitive_{country_tag}_area_weights_debug_{year}.html"
                try:
                    write_d_fugitive_area_weights_debug_map(
                        map_html,
                        bbox_wgs84=area_weights_viz_bbox_wgs84,
                        transform=cor_tr,
                        raster_crs=cor_crs,
                        cams_cells=cams_cells_mask,
                        cell_id=cell_id,
                        corine_l3_121=corine_map_121,
                        corine_l3_123=corine_map_123,
                        corine_l3_131=corine_map_131,
                        population_z=population_z,
                        gem_coal=coal_m,
                        gem_oil=og_m,
                        vnf=vnf_m,
                        osm_by_subgroup=osm_rasters_by_subgroup,
                        W_by_subgroup=W_by_subgroup,
                        W_pollutant=w_poll,
                        legend_alpha_html=legend_alpha_html,
                    )
                    log.info(f"D_Fugitive area-weights debug map: {map_html}")
                except Exception as exc:
                    log.error(f"D_Fugitive area-weights debug map failed: {exc}")
