from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from shapely.geometry import box

from proxy.core import log
from proxy.core.alias import resolve_osm_filepath
from proxy.core.point_matching.fallback import get_corine_airport_facilities
from proxy.core.point_matching.matching import build_aviation_osm_facilities_by_id
from proxy.core.point_matching.sector_flow import run_sector_point_matching
from proxy.dataset_loaders import require_filepaths_exist
from proxy.dataset_loaders.load_cams_points import load_cams_points
from proxy.dataset_loaders.load_osm import load_osm
from proxy.visualizers.viz_map import write_point_match_map
from proxy.writers.point_link import write_cams_facility_link_tif



def build(
    output_dir: Path,
    sector_config_path: Path,
    *,
    sector_config: dict | None = None,
    area_weights: bool = True,
    point_matching: bool = False,
    country_profile: dict[str, str] | None = None,
    crs: str,
    resolution_m: float,
    pad_m: float,
    area_weights_viz_bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> None:
    _ = area_weights_viz_bbox_wgs84

    # 1. FIRST STEP: CHECKING EVERYTHING IS OK WITH THE FILEPATHS
    repo_root = Path(__file__).resolve().parents[3]

    if sector_config is None:
        with sector_config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = sector_config
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
            osm_cfg = cfg["osm"]
            pm = cfg.get("point_matching") if isinstance(cfg.get("point_matching"), dict) else {}

            metric_crs = osm_cfg["metric_crs"]
            clip_buf_m = float(pm.get("osm_clip_buffer_m", 15_000.0))

            lons = [float(r["longitude"]) for r in cams_points.values()]
            lats = [float(r["latitude"]) for r in cams_points.values()]
            pts_m = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326").to_crs(metric_crs)
            minx, miny, maxx, maxy = pts_m.total_bounds
            clip_domain = box(minx - clip_buf_m, miny - clip_buf_m, maxx + clip_buf_m, maxy + clip_buf_m)

            osm_gpkg = repo_root / resolve_osm_filepath(filepaths.get("OSM", {}).get("path"), country_profile)
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
            osm_facilities = build_aviation_osm_facilities_by_id(
                poly_gdf,
                apron_gpkg=apron_gpkg,
                point_matching_cfg=pm,
            )

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

            fallback: dict[str, dict[str, Any]] = {}
            if osm_facilities:
                fallback["osm"] = osm_facilities
            if corine_facilities:
                fallback["corine"] = corine_facilities
            if not fallback:
                raise ValueError("No OSM or CORINE aerodrome facilities for fallback matching.")

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
                        osm_facility_points=osm_facilities,
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
