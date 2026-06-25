from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.warp import Resampling

from proxy.alpha.Compute_alpha_matrix import load_sector_alpha_from_config
from proxy.core import log
from proxy.core.alias import cams_pollutant_var, resolve_osm_filepath
from proxy.core.area_weights import (
    combine_S_solvents_subsectors,
    compute_E_solvents_S_by_activity,
    fuse_alpha_weighted_W_planes,
    normalize_W_per_cams_cell,
)
from proxy.core.point_matching.sector_flow import run_sector_point_matching
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders import require_filepaths_exist
from proxy.core.cams_sector_config import cams_area_emissions, load_sector_cells_mask
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_eprtr_points import load_eprtr_points
from proxy.dataset_loaders.load_osm import load_osm_filtered, rasterize_osm
from proxy.dataset_loaders.load_population import load_population
from proxy.visualizers.area_weights_map import (
    alpha_legend_html,
    write_e_solvents_area_weights_debug_map,
    w_pollutant_for_viz,
)
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.area_weight_stack import (
    area_weights_tif_path,
    open_area_weight_stack,
    write_area_weight_plane,
)
from proxy.writers.point_link import write_cams_facility_link_tif
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
    """GNFR E Solvents: optional CAMS↔E-PRTR links; area weights = activity S, beta subsectors, alpha-fused GeoTIFF."""

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

    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (E_Solvents)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("point_matching needs country_profile from entry")
            raise ValueError("point_matching needs country_profile from entry")

        cps = cfg.get("cams_point_sources") or {}
        year = int(cps.get("year"))
        ec = list(cps.get("emission_category_indices") or [])
        st = list(cps.get("source_type_indices") or [])
        cams_nc = repo_root / str(cams_filepath).replace("\\", "/")
        pol_labels = [str(x).strip() for x in pols if str(x).strip()]
        log.info(
            "Loading CAMS points:"
            f"\n  File     : {cams_filepath}"
            f"\n  Country  : {country_profile['ISO3']}"
            f"\n  Pollutants: {', '.join(pol_labels)}"
        )
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
            eprtr_cfg = cfg.get("eprtr_point_sources") or {}
            reporting_years = [int(y) for y in eprtr_cfg.get("reportingYears")]
            eprtr_sector_raw = eprtr_cfg.get("eprtr_sector_code")
            if eprtr_sector_raw is None:
                raise ValueError("eprtr_point_sources.eprtr_sector_code is required for point matching")
            eprtr_sub_sector_codes = [str(code).strip() for code in eprtr_cfg.get("eprtr_sub_sector_codes")]

            eprtr_points = load_eprtr_points(
                repo_root / str(eprtr_filepath).replace("\\", "/"),
                reporting_years=reporting_years,
                country_full_name=country_profile["full_name"],
                eprtr_sector_code=eprtr_sector_raw,
                eprtr_sub_sector_codes=eprtr_sub_sector_codes,
            )
            log.info(f"Loaded {len(eprtr_points)} E-PRTR facilities")
            fallback = {"eprtr": eprtr_points} if eprtr_points else None
            matches = run_sector_point_matching(
                cams_points,
                cfg=cfg,
                repo_root=repo_root,
                country_profile=country_profile,
                pollutant_labels=pol_labels,
                cams_nc=cams_nc,
                fallback_sources=fallback,
            )

            country_tag = country_profile["full_name"].replace(" ", "_")
            link_tif = write_cams_facility_link_tif(
                matches,
                output_dir / f"E_Solvents_{country_tag}_point_source_{year}.tif",
                crs=crs,
                resolution_m=resolution_m,
                pad_m=pad_m,
            )
            log.info(f"E_Solvents point link GeoTIFF: {link_tif}")

            if log.debug_enabled():
                map_html = output_dir / f"E_Solvents_{country_tag}_point_match_map.html"
                try:
                    write_point_match_map(matches, {}, map_html, eprtr_points=eprtr_points)
                    log.info(f"E_Solvents point match map: {map_html}")
                except Exception as exc:
                    log.error(f"E_Solvents point match map failed: {exc}")

    if area_weights:
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (E_Solvents)")
        log.info("--------------------------------")

        if not country_profile:
            log.error("area_weights needs country_profile from entry")
            raise ValueError("area_weights needs country_profile from entry")

        cps_area = cams_area_emissions(cfg)
        year = int(cps_area["year"])

        corine_cfg = cfg.get("corine") or {}
        corine_band = int(corine_cfg.get("band", 1))

        cams_cells_mask, cams_grid = load_sector_cells_mask(
            repo_root / str(cams_filepath).replace("\\", "/"),
            cfg,
            country_iso3=country_profile["ISO3"],
            pollutants=[str(x).strip() for x in pols if str(x).strip()],
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )

        corine_l3_household = [int(x) for x in (corine_cfg.get("l3_codes_household") or [])]
        corine_l3_service = [int(x) for x in (corine_cfg.get("l3_codes_service") or [])]
        corine_l3_industrial = [int(x) for x in (corine_cfg.get("l3_codes_industrial") or [])]
        corine_l3_transport = [int(x) for x in (corine_cfg.get("l3_codes_transport") or [])]

        corine_map_household, cor_tr, cor_crs, cell_id = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_household,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_service, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_service,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_industrial, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_industrial,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )
        corine_map_transport, _, _, _ = load_corine(
            repo_root / str(corine_filepath).replace("\\", "/"),
            corine_l3_transport,
            corine_band,
            cams_cells_mask,
            cams_grid,
        )

        ch, cw = corine_map_household.shape

        population_map, _, pop_transform, pop_crs, pop_nodata = load_population(
            repo_root / str(population_filepath).replace("\\", "/"),
            cams_cells_mask,
        )
        if population_map is None:
            log.warning("E_Solvents area_weights: no population raster; household uses CORINE only.")
            population_z = np.zeros((ch, cw), dtype=np.float32)
        else:
            population_map = warp_raster_to_grid(
                population_map, pop_transform, pop_crs, ch, cw, cor_tr, cor_crs,
                src_nodata=pop_nodata, dest_init_nan=True,
            )
            inside_z = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells_mask) & np.isfinite(
                population_map
            )
            population_z = z_score_inside(
                population_map,
                inside_z,
                upper_quantile=0.99,
                rescale_to_01=True,
            ).astype(np.float32)

        if not cams_cells_mask:
            log.warning("E_Solvents area_weights: no CAMS cells; skipping.")
            return

        osm_cfg = cfg.get("osm") or {}
        sub_osm = osm_cfg.get("subgroups") or {}
        osm_gpkg = repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile)
        rz = osm_cfg.get("rasterize") or {}
        osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]] = {}

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
                osm_r = rasterize_osm(gdf, ch, cw, cor_tr, cor_crs, rz, cams_cells_mask)
                osm_rasters_by_subgroup[sg_name][sid] = np.asarray(
                    osm_r, dtype=np.float32, order="C", copy=True
                )
                log.info(
                    f"E_Solvents OSM subgroup={sg_name} slot={sid} "
                    f"features={len(gdf)} grid_sum={float(np.sum(osm_r)):.6g}"
                )

        weights_cfg = cfg.get("weights") or {}
        S_by_activity = compute_E_solvents_S_by_activity(
            osm_rasters_by_subgroup=osm_rasters_by_subgroup,
            corine_map_household=corine_map_household,
            corine_map_service=corine_map_service,
            corine_map_industrial=corine_map_industrial,
            corine_map_transport=corine_map_transport,
            population_z=population_z,
            weights_cfg=weights_cfg,
        )
        for act, S_a in S_by_activity.items():
            log.info(f"E_Solvents activity={act} S_sum={float(np.sum(S_a)):.6g}")

        beta_cfg = cfg.get("beta")
        if not isinstance(beta_cfg, dict) or not beta_cfg:
            raise ValueError("sector config beta block missing or empty")

        alpha_sector_key = "E_Solvents"
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
            f"{alpha_sector_key} alpha shape {alpha_result.alpha.shape} "
            f"(pollutants x groups); methods={alpha_result.methods.tolist()}"
        )
        for j, plab in enumerate(alpha_result.pollutant_labels):
            parts = [
                f"{group_names[i]}={float(alpha_result.alpha[j, i]):.4f}"
                for i in range(len(group_names))
            ]
            log.info(f"  {plab}: " + " ".join(parts) + f" (method {int(alpha_result.methods[j])})")

        missing_beta = [g for g in group_names if g not in beta_cfg]
        if missing_beta:
            raise ValueError(f"beta missing rows for alpha groups: {missing_beta}")

        activity_keys = list(weights_cfg.keys())
        make_viz = log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None
        S_for_viz = (
            {k: np.asarray(v, dtype=np.float32) for k, v in S_by_activity.items()} if make_viz else None
        )

        S_by_subsector = combine_S_solvents_subsectors(
            S_by_activity,
            beta_cfg,
            activity_keys=activity_keys,
            subsector_order=group_names,
        )
        del S_by_activity
        gc.collect()

        W_by_subsector: dict[str, np.ndarray] = {}
        for sname, S_s in list(S_by_subsector.items()):
            W_s = normalize_W_per_cams_cell(S_s, cell_id, cams_cells_mask)
            W_by_subsector[sname] = W_s
            log.info(
                f"E_Solvents subsector={sname} S_sum={float(np.sum(S_s)):.6g} "
                f"W_sum={float(np.sum(W_s)):.6g}"
            )
            del S_s
        S_by_subsector.clear()
        del S_by_subsector
        gc.collect()

        country_tag = country_profile["full_name"].replace(" ", "_")
        maybe_export_w_groups(
            export_w_groups,
            w_groups_export_root,
            sector_key="E_Solvents",
            country_tag=country_tag,
            year=year,
            W_by_group=W_by_subsector,
            cell_id=cell_id,
            transform=cor_tr,
            crs=cor_crs,
            alpha_result=alpha_result,
            cams_cells=cams_cells_mask,
        )

        W_per_group = [W_by_subsector[g] for g in group_names]

        a_alpha = alpha_result.alpha.astype(np.float32)
        n_poll = int(a_alpha.shape[0])
        n_g = len(W_per_group)
        if a_alpha.shape[1] != n_g:
            raise ValueError(f"alpha columns {a_alpha.shape[1]} != subsector groups {n_g}")

        band_names = [cams_pollutant_var(x) for x in alpha_result.pollutant_labels]
        out_w_tif = area_weights_tif_path(output_dir, "E_Solvents", country_tag, year)
        dst = open_area_weight_stack(out_w_tif, ch, cw, n_poll, cor_tr, cor_crs)
        W_poll_stack = np.zeros((n_poll, ch, cw), dtype=np.float32)
        for j in range(n_poll):
            W_c = fuse_alpha_weighted_W_planes(
                W_per_group, a_alpha[j], cell_id, cams_cells_mask,
            )
            W_poll_stack[j] = W_c
            write_area_weight_plane(dst, j + 1, band_names[j], W_c)

        dst.close()
        w_poll = w_pollutant_for_viz(cfg, W_poll_stack, alpha_result.pollutant_labels) if make_viz else {}
        del W_per_group
        if not make_viz:
            del W_by_subsector
        gc.collect()
        log.info(f"E_Solvents alpha-fused area weights GeoTIFF: {out_w_tif}")

        if make_viz and S_for_viz is not None:
            legend_alpha_html = alpha_legend_html(
                alpha_result.pollutant_labels,
                alpha_result.alpha,
                group_names,
                cfg,
            )
            map_html = output_dir / f"E_Solvents_{country_tag}_area_weights_debug_{year}.html"
            try:
                write_e_solvents_area_weights_debug_map(
                    map_html,
                    bbox_wgs84=area_weights_viz_bbox_wgs84,
                    transform=cor_tr,
                    raster_crs=cor_crs,
                    cams_cells=cams_cells_mask,
                    cell_id=cell_id,
                    S_archetype=S_for_viz,
                    W_pollutant=w_poll,
                    legend_alpha_html=legend_alpha_html,
                )
                log.info(f"E_Solvents area-weights debug map: {map_html}")
            except Exception as exc:
                log.error(f"E_Solvents area-weights debug map failed: {exc}")
            finally:
                del S_for_viz, W_by_subsector
                gc.collect()
