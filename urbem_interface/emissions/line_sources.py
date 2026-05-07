"""
Line sources pipeline — replicate
`code/UrbEm/UrbEm_v1.0_Rscripts/3_UrbEm_linesources_CAMS8.1_Ioannina_v3.R`.

High-level:
  - Read CAMS area emissions for traffic sectors (F1–F4)
  - Project to domain, optional population proxy, optional GHSL urban-centre upweight
  - Distribute area emissions onto OSM road network (VEIN-inspired)
  - Mass correction, road widths, UECT line-source CSV
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling as WarpResampling

from urbem_interface.logging_config import configure_urbem_logging, get_logger
from urbem_interface.utils.config_loader import resolve_paths, load_run_config, load_linesources_config, load_proxies_config
from urbem_interface.emissions.prepare_cams import prepare_cams_emissions, POLLUTANT_NAMES
from urbem_interface.emissions.proxy_preparation import proxy_cwd, proxy_distribution
from urbem_interface.utils.grid import _compute_proxy_coarse_grid, _load_and_warp_proxy, _warp_to_domain
from urbem_interface.emissions.line_utils import (
    area_to_osm_lines,
    domain_bounds_wgs84_from_cfg,
    fetch_osm_roads,
    line_start_end_coords,
    POLLUTANTS_OUT,
)
from urbem_interface.emissions.uect import UECT_POLLUTANTS_ORDER
from urbem_interface.emissions.area_sources import _save_raster_stack_csv

logger = get_logger(__name__)

KGYR_TO_GS = 1e3 / (365 * 24 * 3600)


def run_line_sources(
    run_config: dict,
    lines_config: dict,
    proxies_config: dict,
    paths: dict,
    config_dir: Path,
    intermediates_dir: Path | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run the full line sources pipeline. Returns UECT line DataFrame.
    """
    configure_urbem_logging(debug=run_config.get("debug"))

    domain_cfg = run_config["domain"]
    year = int(run_config["year"])
    domain_crs = CRS.from_string(domain_cfg["crs"])
    nrow = int(domain_cfg["nrow"])
    ncol = int(domain_cfg["ncol"])
    xmin = float(domain_cfg["xmin"])
    ymin = float(domain_cfg["ymin"])
    xmax = float(domain_cfg["xmax"])
    ymax = float(domain_cfg["ymax"])
    crs_str = domain_cfg["crs"]
    domain_transform = from_bounds(xmin, ymin, xmax, ymax, ncol, nrow)
    domain_shape = (nrow, ncol)
    domain_arr = np.ones(domain_shape, dtype=np.float64)
    wgs84 = CRS.from_epsg(4326)

    gnfr_list = lines_config.get("gnfr_sectors", [])
    if not gnfr_list:
        gnfr_list = [
            "F1_RoadTransport_Exhaust_Gasoline",
            "F2_RoadTransport_Exhaust_Diesel",
            "F3_RoadTransport_Exhaust_LPG_gas",
            "F4_RoadTransport_NonExhaust",
        ]

    pollutants_in = list(POLLUTANT_NAMES.keys())
    poll_names_out = [POLLUTANT_NAMES[k] for k in pollutants_in]

    cams_folder = paths["cams_folder"]
    proxies_folder = paths["proxies_folder"]
    nc_files = sorted(Path(cams_folder).glob(f"*{year}.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF for year {year} in {cams_folder}")
    nc_path = nc_files[0]

    logger.info("━━━ [1/8] Reading CAMS area emissions (F1–F4) ━━━━━━━━━━━━━")
    cams_result = prepare_cams_emissions(
        nc_path,
        source_type="area",
        pollutants=pollutants_in,
        sectors=gnfr_list,
    )

    cams_lon = cams_result.grid_lons
    cams_lat = cams_result.grid_lats
    nlat, nlon = len(cams_lat), len(cams_lon)
    xmin_wgs = float(cams_lon.min())
    xmax_wgs = float(cams_lon.max())
    ymin_wgs = float(cams_lat.min())
    ymax_wgs = float(cams_lat.max())
    cams_transform_wgs = from_bounds(xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs, nlon, nlat)

    def emit(stage_id: str):
        if progress_callback:
            progress_callback(stage_id, {})

    logger.info("━━━ [2/8] Projecting CAMS to domain ━━━━━━━━━━━━━━━━━━━━━")
    cams_stacked: dict[str, np.ndarray] = {p: np.zeros(domain_shape, dtype=np.float64) for p in poll_names_out}
    for sector_name in gnfr_list:
        if sector_name not in cams_result.rasters:
            continue
        sector_data = cams_result.rasters[sector_name]
        for poll, arr in sector_data.items():
            warped = _warp_to_domain(
                arr, cams_transform_wgs, wgs84,
                domain_transform, domain_shape, domain_crs,
                domain_bounds=(xmin, ymin, xmax, ymax),
                resampling="nearest",
                label=f"{sector_name}-{poll}",
            )
            cams_stacked[poll] = cams_stacked[poll] + np.where(np.isnan(warped), 0.0, warped)

    logger.info(f"  summed {len(gnfr_list)} sectors to domain grid")
    emit("cams")
    if intermediates_dir:
        line_cams_dir = intermediates_dir / "line_cams"
        line_cams_dir.mkdir(parents=True, exist_ok=True)
        _save_raster_stack_csv(
            cams_stacked,
            line_cams_dir / "cams_stacked.csv",
            nrow, ncol,
            poll_names=poll_names_out,
        )
        logger.debug(f"  saved line CAMS grid -> {line_cams_dir}")

    pop_proxy = str(lines_config.get("pop_proxy", "yes")).lower() == "yes"
    if pop_proxy:
        logger.info("━━━ [3/8] Population proxy downscaling ━━━━━━━━━━━━━━━━━")
        cams_origin_arr, cams_origin_transform, nlon_c, nlat_c, coarse_bounds = _compute_proxy_coarse_grid(
            src_shape=(nlat, nlon),
            src_transform=cams_transform_wgs,
            src_crs=wgs84,
            dst_crs=domain_crs,
            domain_bounds=(xmin, ymin, xmax, ymax),
            res_x=7000.0,
            res_y=5550.0,
        )
        pop_file = proxies_config.get("proxies", {}).get("population")
        if pop_file:
            pop_path = proxies_folder / pop_file
            if pop_path.exists():
                pop_arr = _load_and_warp_proxy(
                    pop_path, domain_transform, domain_shape, domain_crs,
                    resampling=WarpResampling.bilinear,
                )
                pop_norm = proxy_cwd(
                    cams_origin_arr, cams_origin_transform, domain_crs,
                    domain_arr, domain_transform, domain_crs,
                    pop_arr, domain_transform, domain_crs,
                    sparse_threshold=0.05,
                    max_weight_per_cell=0.5,
                )
                cams_stacked = proxy_distribution(cams_stacked, pop_norm, "coarse_cells_proxy")
                logger.info("  population proxy applied")
            else:
                logger.warning(f"  population proxy file not found: {pop_path}")
        else:
            logger.warning("  no population proxy in config")
    else:
        logger.info("━━━ [3/8] Skipping population proxy ━━━━━━━━━━━━━━━━━")
    emit("proxies")

    centre = str(lines_config.get("centre", "yes")).lower() == "yes"
    if centre:
        logger.info("━━━ [4/8] GHSL urban-centre upweighting ━━━━━━━━━━━━━━━")
        uc_factor = float(lines_config.get("centre_factor", 3))
        centre_pollutants = str(lines_config.get("centre_pollutants", "all")).lower()
        ghsl_file = proxies_config.get("ghsl_urbancentre", "ghs_europe_iso3.tif")
        ghsl_path = proxies_folder / ghsl_file
        if ghsl_path.exists():
            ghsl_arr = _load_and_warp_proxy(
                ghsl_path, domain_transform, domain_shape, domain_crs,
                resampling=WarpResampling.nearest,
            )
            uc_mult = np.where(ghsl_arr > 0, uc_factor, 1.0)
            if centre_pollutants == "all":
                for poll in cams_stacked:
                    cams_stacked[poll] = cams_stacked[poll] * uc_mult
            else:
                if "NOx" in cams_stacked:
                    cams_stacked["NOx"] = cams_stacked["NOx"] * uc_mult
            logger.info(f"  GHSL centre factor={uc_factor} applied")
        else:
            logger.warning(f"  GHSL file not found: {ghsl_path}")
    else:
        logger.info("━━━ [4/8] Skipping GHSL centre ━━━━━━━━━━━━━━━━━━━━━")

    if intermediates_dir:
        line_down_dir = intermediates_dir / "line_downscaled"
        line_down_dir.mkdir(parents=True, exist_ok=True)
        _save_raster_stack_csv(
            cams_stacked,
            line_down_dir / "cams_stacked.csv",
            nrow, ncol,
            poll_names=poll_names_out,
        )
        logger.debug(f"  saved line downscaled grid -> {line_down_dir}")

    logger.info("━━━ [5/8] Fetching OSM roads & distributing to lines ━━━━━")
    bbox_wgs = domain_bounds_wgs84_from_cfg(domain_cfg)
    road_types = lines_config.get("road_types", [
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
    ])
    road_type_weights = lines_config.get("road_type_weights", {
        "motorway": 10, "motorway_link": 10,
        "trunk": 5, "trunk_link": 5,
        "primary": 2, "primary_link": 2,
        "secondary": 2, "secondary_link": 2,
    })

    osm_gdf = fetch_osm_roads(bbox_wgs, road_types, domain_crs)
    emit("osm")
    if osm_gdf.empty:
        logger.warning("  No OSM roads — returning empty line table")
        return _empty_uect_lines()

    emissions_for_lines = {p: cams_stacked.get(p, np.zeros(domain_shape)) for p in POLLUTANTS_OUT}
    lines_gdf = area_to_osm_lines(
        domain_transform, domain_shape, domain_crs,
        emissions_for_lines, osm_gdf, road_type_weights,
        split_by_cell=bool(str(lines_config.get("split_by_cell", "true")).lower() in ("1", "true", "yes")),
        road_grouping=str(lines_config.get("road_grouping", "r4")),
        include_zero_emissions_cells=bool(
            str(lines_config.get("include_zero_emissions_cells", "true")).lower() in ("1", "true", "yes")
        ),
    )

    if lines_gdf.empty:
        logger.warning("  No line segments after allocation")
        return _empty_uect_lines()

    emit("lines")
    logger.info("━━━ [6/8] Mass correction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for poll in POLLUTANTS_OUT:
        if poll not in lines_gdf.columns:
            continue
        area_sum = float(np.nansum(cams_stacked.get(poll, np.zeros(domain_shape))))
        line_sum = float(lines_gdf[poll].sum())
        if line_sum > 0 and area_sum > 0:
            lines_gdf[poll] = lines_gdf[poll] * (area_sum / line_sum)

    logger.info("━━━ [7/8] Road widths & start/end coords ━━━━━━━━━━━━━━━")
    road_width_m = lines_config.get("road_width_m", {})
    default_width = 8.0
    lines_gdf["width"] = lines_gdf["roadtype"].apply(
        lambda h: road_width_m.get(str(h), road_width_m.get("tertiary", default_width))
    )

    x_start, y_start, x_end, y_end = line_start_end_coords(lines_gdf)

    logger.info("━━━ [8/8] Building UECT line table ━━━━━━━━━━━━━━━━━━━━")
    snap = int(lines_config.get("snap", 7))
    uect = pd.DataFrame({
        "snap": snap,
        "xcor_start": x_start.astype(int),
        "ycor_start": y_start.astype(int),
        "xcor_end": x_end.astype(int),
        "ycor_end": y_end.astype(int),
        "elevation": 0,
        "width": lines_gdf["width"].values,
        "roadtype": lines_gdf["roadtype"].values,
    })
    for poll in UECT_POLLUTANTS_ORDER:
        if poll in lines_gdf.columns:
            uect[poll] = lines_gdf[poll].values * KGYR_TO_GS
        else:
            uect[poll] = 0.0
    uect["PN"] = np.nan
    uect = uect.replace({np.nan: -999})

    logger.info(f"  {len(uect)} line segments")
    return uect


def _empty_uect_lines() -> pd.DataFrame:
    cols = ["snap", "xcor_start", "ycor_start", "xcor_end", "ycor_end", "elevation", "width", "roadtype"]
    cols += list(UECT_POLLUTANTS_ORDER) + ["PN"]
    return pd.DataFrame(columns=cols)


def run_and_export(
    run_config_path: Path,
    lines_config_path: Path,
    proxies_config_path: Path,
    progress_callback=None,
) -> Path:
    run_config = load_run_config(run_config_path)
    configure_urbem_logging(debug=run_config.get("debug"))

    lines_config = load_linesources_config(lines_config_path)

    config_dir = run_config_path.parent
    paths = resolve_paths(run_config, config_dir)
    proxies_config = load_proxies_config(
        proxies_config_path, proxies_folder=paths["proxies_folder"]
    )

    out_folder = paths["emission_output_folder"]
    out_folder.mkdir(parents=True, exist_ok=True)

    intermediates_dir = out_folder / "intermediates"
    uect_df = run_line_sources(
        run_config, lines_config, proxies_config, paths, config_dir,
        intermediates_dir=intermediates_dir,
        progress_callback=progress_callback,
    )

    region = run_config.get("region", "Ioannina")
    stem = f"{region}_linesources_CAMS-REG-AP_8.1"
    out_path = out_folder / f"{stem}_{len(uect_df)}_lsrc.csv"
    uect_df.to_csv(out_path, index=False, na_rep="-999")

    return out_path
