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
    combined_S_industry_group,
    fuse_alpha_weighted_W_planes,
    normalize_W_per_cams_cell,
)
from proxy.core.point_matching.sector_flow import run_sector_point_matching
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_eprtr_points import load_eprtr_points
from proxy.dataset_loaders.load_osm import load_osm_filtered, rasterize_osm
from proxy.dataset_loaders.load_population import load_population
from proxy.visualizers.area_weights_map import (
    write_b_industry_area_weights_debug_map,
    w_pollutant_for_viz,
)
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.area_weight_stack import area_weights_tif_path, write_area_weight_stack_multiband
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
    """GNFR B industry: optional CAMS↔E-PRTR point links; area weights = OSM+CORINE+pop per group, alpha-fused GeoTIFF."""

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

    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (B_Industry)")
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
                output_dir / f"B_Industry_{country_tag}_point_source_{year}.tif",
                crs=crs,
                resolution_m=resolution_m,
                pad_m=pad_m,
            )
            log.info(f"B_Industry point link GeoTIFF: {link_tif}")

            if log.debug_enabled():
                map_html = output_dir / f"B_Industry_{country_tag}_point_match_map.html"
                try:
                    write_point_match_map(
                        matches,
                        {},
                        map_html,
                        eprtr_points=eprtr_points,
                    )
                    log.info(f"B_Industry point match map (open in browser): {map_html}")
                except Exception as exc:
                    log.error(f"B_Industry point match map failed: {exc}")

    if area_weights:
        # Raster stack on CORINE reference grid: four inventory groups share one alpha axis
        # (alpha_methods.yaml). Per group: CORINE L3 mask, OSM subset (industrial_layer), population z-score,
        # S = (1-w_pop)*(w_osm*OSM + w_clc*CORINE) + w_pop*Z_pop, then W = normalize S inside each CAMS cell.
        log.info("--------------------------------")
        log.info("AREA WEIGHTS (B_Industry)")
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
        if not population_filepath:
            raise ValueError("filepaths.Population.path is required for B_Industry area weights")

        alpha_sector_key = "B_Industry"
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
            f"(pollutants × groups); methods={alpha_result.methods.tolist()}"
        )
        for j, plab in enumerate(alpha_result.pollutant_labels):
            parts = [
                f"{alpha_result.group_names[i]}={float(alpha_result.alpha[j, i]):.4f}"
                for i in range(len(alpha_result.group_names))
            ]
            log.info(f"  {plab}: " + " ".join(parts) + f" (method {int(alpha_result.methods[j])})")

        weights_cfg = cfg.get("weights") or {}
        osm_cfg = cfg["osm"]
        osm_groups = osm_cfg.get("groups") or {}
        if not isinstance(osm_groups, dict) or not osm_groups:
            raise ValueError("osm.groups must be a non-empty mapping with one entry per inventory group")
        for gx in group_names:
            if gx not in osm_groups:
                raise ValueError(
                    f"osm.groups missing key {gx!r}; use the same names as alpha_methods.yaml / weights / corine"
                )
        layer_map = osm_cfg.get("layer_map") or {}
        layer_order = list(osm_cfg.get("layer_order") or [])
        if isinstance(layer_map, dict) and layer_map and layer_order:
            want = {layer_map.get("point"), layer_map.get("line"), layer_map.get("polygon")}
            if want != {None} and None not in want and set(layer_order) != want:
                raise ValueError(
                    f"osm.layer_order values {set(layer_order)!r} must equal osm.layer_map targets {want!r}"
                )

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
            log.warning("B_Industry area_weights: no CAMS area cells for this filter; skipping raster stack.")
        else:
            cell_id: np.ndarray | None = None
            cor_tr = None
            cor_crs = None
            ch = cw = 0
            # Reference CORINE L3 121 and 131 on the same grid (for DEBUG Folium overlays only).
            cor121_map: np.ndarray | None = None
            cor131_map: np.ndarray | None = None
            osm_by_group: dict[str, np.ndarray] = {}
            W_by_group: dict[str, np.ndarray] = {}
            mix_by_group: dict[str, dict] = {}

            W_per_group: list[np.ndarray] = []

            for g in group_names:
                wrow = weights_cfg.get(g) or {}
                if not wrow:
                    raise ValueError(f"weights.{g} is missing for B_Industry area weights")
                w_osm = float(wrow["w_osm"])
                w_clc = float(wrow["w_clc"])
                w_pop = float(wrow["w_pop"])

                cor_sub_cfg = corine_cfg.get(g) or {}
                l3_codes = [int(x) for x in (cor_sub_cfg.get("l3_codes") or [])]
                cor_map, tr, crs, cid = load_corine(
                    repo_root / str(corine_filepath).replace("\\", "/"),
                    l3_codes,
                    corine_band,
                    cams_cells_mask,
                    cams_grid,
                )
                if cell_id is None:
                    cell_id = cid
                    cor_tr, cor_crs = tr, crs
                    ch, cw = cor_map.shape
                    if log.debug_enabled():
                        cor121_map, _, _, _ = load_corine(
                            repo_root / str(corine_filepath).replace("\\", "/"),
                            [121],
                            corine_band,
                            cams_cells_mask,
                            cams_grid,
                        )
                        cor131_map, _, _, _ = load_corine(
                            repo_root / str(corine_filepath).replace("\\", "/"),
                            [131],
                            corine_band,
                            cams_cells_mask,
                            cams_grid,
                        )
                        if cor121_map.shape != (ch, cw) or cor131_map.shape != (ch, cw):
                            raise ValueError("CORINE 121/131 debug rasters must match group CORINE grid shape")
                
                gspec = osm_groups[g]
                if not isinstance(gspec, dict):
                    raise TypeError(f"osm.groups.{g} must be a mapping")
                inc = gspec.get("industrial_layer_include")
                ids = frozenset(str(x) for x in inc)
                buf_ov = gspec.get("buffer_m")
    
                osm_gdf = load_osm_filtered(
                    repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile),
                    cams_cells_mask,
                    osm_cfg,
                    column="industrial_layer",
                    values=ids,
                    buffer_m_override=buf_ov if isinstance(buf_ov, dict) else None,
                )
                osm_r = rasterize_osm(
                    osm_gdf,
                    ch,
                    cw,
                    cor_tr,
                    cor_crs,
                    osm_cfg["rasterize"],
                    cams_cells_mask,
                )

                if not W_per_group:
                    population_map, _, pop_transform, pop_crs, pop_nodata = load_population(
                        repo_root / str(population_filepath).replace("\\", "/"),
                        cams_cells_mask,
                    )
                    population_map = warp_raster_to_grid(
                        population_map, pop_transform, pop_crs, ch, cw, cor_tr, cor_crs,
                        src_nodata=pop_nodata, dest_init_nan=True, nan_fill=0.0,
                    )

                    inside_pop = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells_mask) & np.isfinite(
                        population_map
                    )
                    population_z = z_score_inside(
                        population_map,
                        inside_pop,
                        upper_quantile=0.99,
                        rescale_to_01=True,
                    )
                assert cell_id is not None

                cor_f = cor_map.astype(np.float64)
                osm_f = osm_r.astype(np.float64)
                S_g = combined_S_industry_group(
                    osm_f,
                    cor_f,
                    population_z,
                    w_osm=w_osm,
                    w_clc=w_clc,
                    w_pop=w_pop,
                )
                W_g = normalize_W_per_cams_cell(S_g, cell_id, cams_cells_mask)
                osm_by_group[g] = np.asarray(osm_r, dtype=np.float32, order="C", copy=True)
                W_by_group[g] = np.asarray(W_g, dtype=np.float64, order="C", copy=True)
                mix_by_group[g] = {
                    "mixer": "industry_group",
                    "weights": {"w_osm": w_osm, "w_clc": w_clc, "w_pop": w_pop},
                    "terms": {
                        "osm": np.asarray(osm_f, dtype=np.float32),
                        "corine": np.asarray(cor_f, dtype=np.float32),
                        "pop_z": np.asarray(population_z, dtype=np.float32),
                    },
                }
                W_per_group.append(W_g)
                log.info(
                    f"  group {g}: CORINE L3 {l3_codes} OSM layers={len(ids)} ids | "
                    f"W sum over grid={float(np.sum(W_g)):.6g}"
                )

            a_alpha = alpha_result.alpha.astype(np.float64)
            n_poll = a_alpha.shape[0]
            n_g = len(W_per_group)
            if a_alpha.shape[1] != n_g:
                raise ValueError(f"alpha columns {a_alpha.shape[1]} != spatial groups {n_g}")

            country_tag = country_profile["full_name"].replace(" ", "_")
            maybe_export_w_groups(
                export_w_groups,
                w_groups_export_root,
                sector_key="B_Industry",
                country_tag=country_tag,
                year=year,
                W_by_group=W_by_group,
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
            out_w_tif = area_weights_tif_path(output_dir, "B_Industry", country_tag, year)
            write_area_weight_stack_multiband(
                out_w_tif,
                W_poll_stack,
                band_names,
                cor_tr,
                cor_crs,
            )
            log.info(f"B_Industry alpha-fused area weights GeoTIFF: {out_w_tif}")

            if log.debug_enabled() and area_weights_viz_bbox_wgs84 is not None:
                if cor121_map is None or cor131_map is None:
                    log.warning("B_Industry area-weights debug map: CORINE 121/131 rasters missing; skip.")
                else:
                    w_poll = w_pollutant_for_viz(cfg, W_poll_stack, alpha_result.pollutant_labels)
                    map_html = output_dir / f"B_Industry_{country_tag}_area_weights_debug_{year}.html"
                    try:
                        write_b_industry_area_weights_debug_map(
                            map_html,
                            bbox_wgs84=area_weights_viz_bbox_wgs84,
                            transform=cor_tr,
                            raster_crs=cor_crs,
                            cams_cells=cams_cells_mask,
                            cell_id=cell_id,
                            corine_l3_121=cor121_map,
                            corine_l3_131=cor131_map,
                            osm_by_group=osm_by_group,
                            W_by_group=W_by_group,
                            W_pollutant=w_poll,
                        )
                        log.info(f"B_Industry area-weights debug map: {map_html}")
                    except Exception as exc:
                        log.error(f"B_Industry area-weights debug map failed: {exc}")
