from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from shapely.geometry import box

from proxy.core import log
from proxy.core.alias import resolve_osm_filepath
from proxy.core.point_matching.fallback import (
    get_corine_airport_facilities,
    merge_corine_aviation_fallback,
)
from proxy.core.point_matching.matching import (
    build_aviation_osm_facilities_by_id,
    match_cams_to_facilities_one_to_one,
    point_match_settings,
)
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_osm import load_osm
from proxy.visualizers.viz_map import write_point_match_map
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
    export_w_groups: bool = False,
    w_groups_export_root: Path | None = None,
) -> None:
    _ = (export_w_groups, w_groups_export_root)
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

    if point_matching:
        log.info("--------------------------------")
        log.info("POINT MATCHING (H_Aviation)")
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
            log.debug(f"Max match distance: {max_match_distance_km} km")
        else:
            log.info("Point matching mode: same CAMS cell")

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

        osm_cfg = cfg["osm"]
        pm = cfg.get("point_matching") if isinstance(cfg.get("point_matching"), dict) else {} 

        # Here we create our own grid instead of using the CAMS one because in some countries there is no CAMS area source data for aviation.
        metric_crs = osm_cfg["metric_crs"]
        clip_buf_m = float(pm.get("osm_clip_buffer_m", 15_000.0))

        lons = [float(r["longitude"]) for r in cams_points.values()]
        lats = [float(r["latitude"]) for r in cams_points.values()]
        pts_m = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326").to_crs(metric_crs)
        minx, miny, maxx, maxy = pts_m.total_bounds
        clip_domain = box(minx - clip_buf_m, miny - clip_buf_m, maxx + clip_buf_m, maxy + clip_buf_m)
        log.debug(
            f"OSM clip domain ({metric_crs}): bbox expanded by {clip_buf_m:.0f} m around CAMS point footprint"
        )

        osm_gpkg = repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile)
        # Here we load the OSM polygons and clip them to the CAMS country footprint.
        poly_gdf = load_osm(
            osm_gpkg,
            None,
            osm_cfg,
            for_point_matching=True,
            clip_domain_metric=clip_domain,
        )
        log.info(f"OSM aerodrome polygons (clipped, unbuffered): {len(poly_gdf)} rows")

        rel_apron = str(pm.get("aviation_terminal_apron_gpkg") or "").strip()
        apron_gpkg = (
            (Path(rel_apron) if Path(rel_apron).is_absolute() else repo_root / rel_apron).resolve()
            if rel_apron
            else osm_gpkg.resolve()
        )
        facilities = build_aviation_osm_facilities_by_id(
            poly_gdf,
            apron_gpkg=apron_gpkg,
            point_matching_cfg=pm,
        )
        if not facilities:
            raise ValueError(
                "No OSM aerodrome polygon candidates after filters; check GPKG and point_matching config."
            )

        # Implementing the point matching.
        matches = match_cams_to_facilities_one_to_one(
            cams_points,
            facilities,
            match_mode=match_mode,
            max_match_distance_km=max_match_distance_km,
            cams_grid_meta=cams_grid_meta,
            facility_id_field_in_output_rows="osm_facility_id",
            facility_info_field_in_output_rows="osm_facility_info",
            log_label_for_facility_dataset="OSM aerodrome",
        )
        n_osm = sum(1 for m in matches.values() if m.get("matched") == "yes")
        unmatched = {pid: dict(matches[pid]["cams"]) for pid in matches if matches[pid].get("matched") != "yes"}
        corine_facilities: dict[str, dict[str, Any]] = {}
        
        # check if we have non-matched CAMS points and if we do, we need to load the CORINE data and match the CAMS points to the CORINE data.
        if unmatched:
            corine_cfg = cfg.get("corine") or {}
            corine_band = int(corine_cfg.get("band", 1))
            corine_l3_codes = [int(x) for x in (corine_cfg.get("l3_codes") or [])]
            min_patch_m2 = float(corine_cfg.get("min_patch_area_m2"))
       
            corine_facilities = get_corine_airport_facilities(
                repo_root / str(corine_filepath).replace("\\", "/"),
                corine_band,
                corine_l3_codes,
                clip_domain,
                metric_crs,
                min_patch_area_m2=min_patch_m2,
            )
            if corine_facilities:
                cor_fb = match_cams_to_facilities_one_to_one(
                    unmatched,
                    corine_facilities,
                    match_mode=match_mode,
                    max_match_distance_km=max_match_distance_km,
                    cams_grid_meta=cams_grid_meta,
                    facility_id_field_in_output_rows="corine_facility_id",
                    facility_info_field_in_output_rows="corine_facility_info",
                    log_label_for_facility_dataset="CORINE airport",
                )
                merge_corine_aviation_fallback(matches, cor_fb)

        n_yes = sum(1 for m in matches.values() if m.get("matched") == "yes")
        n_cor = sum(1 for m in matches.values() if m.get("match_source") == "corine")
        log.info(
            f"H_Aviation point matching: {n_osm}/{len(matches)} CAMS→OSM, "
            f"{n_cor} via CORINE, {n_yes} total matched ."
        )

        country_tag = country_profile["full_name"].replace(" ", "_")
        link_tif = write_cams_facility_link_tif(
            matches,
            output_dir / f"H_Aviation_{country_tag}_point_source_{year}.tif",
            crs=crs,
            resolution_m=resolution_m,
            pad_m=pad_m,
        )
        log.info(f"H_Aviation point link GeoTIFF: {link_tif}")

        if log.debug_enabled():
            map_html = output_dir / f"H_Aviation_{country_tag}_point_match_map.html"
            try:
                write_point_match_map(
                    matches,
                    {},
                    map_html,
                    eprtr_points=None,
                    osm_facility_points=facilities,
                    osm_polygons_gdf=poly_gdf,
                    corine_facility_points=corine_facilities if corine_facilities else None,
                )
                log.info(f"H_Aviation point match map (open in browser): {map_html}")
            except Exception as exc:
                log.error(f"H_Aviation point match map failed: {exc}")

    if area_weights:
        log.info(
            "H_Aviation: area weights are not implemented (EU-only scope). "
            "CAMS-REG v8 has no GNFR H area-source parents for EU countries; "
            "aviation downscaling here is point-source matching only."
        )

    if not point_matching and not area_weights:
        log.info("H_Aviation: point_matching and area_weights disabled; nothing to run.")
