"""
Point sources pipeline — replicate
`code/UrbEm/UrbEm_v1.0_Rscripts/1_UrbEm_pointsources_CAMS8.1_Ioannina_v2.R`.

High-level:
  - Read CAMS point-source rasters (WGS84 grid, by sector/pollutant)
  - Read RI-URBANS point inventory (real coordinates)
  - Distribute CAMS point emissions to RI-URBANS points per CAMS cell (weighted)
  - Unmatched CAMS point emissions are added back into area-source rasters
  - Write UECT point-source CSV + corrected area-sources artifact (.npz)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling as WarpResampling

from urbem_interface.logging_config import configure_urbem_logging, get_logger
from urbem_interface.utils.config_loader import resolve_paths, load_run_config, load_pointsources_config, load_snap_mapping
from urbem_interface.emissions.prepare_cams import prepare_cams_emissions, POLLUTANT_NAMES
from urbem_interface.emissions.uect import UECT_POLLUTANTS_ORDER, UectPointDefaults
from urbem_interface.emissions.point_utils import (
    assign_cams_cell_indices,
    coarse_grid_from_cams,
    domain_bounds_wgs84,
    project_points_to_domain,
    rasterize_unmatched_to_coarse,
)


logger = get_logger(__name__)


def run_point_sources(
    run_config: dict,
    points_config: dict,
    snap_config: dict,
    paths: dict,
    config_dir: Path,
    intermediates_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Returns:
      points_df (UECT formatted, before dropping zeros)
      corrected_area (sector -> pollutant -> domain array)
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
    domain_transform = from_bounds(xmin, ymin, xmax, ymax, ncol, nrow)
    domain_shape = (nrow, ncol)

    cams_folder = paths["cams_folder"]
    nc_files = sorted(Path(cams_folder).glob(f"*{year}.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF for year {year} in {cams_folder}")
    nc_path = nc_files[0]

    logger.info("━━━ [1/6] Loading CAMS point + area emissions ━━━━━━━━━━━")
    gnfr_list = list(snap_config.get("gnfr_to_snap", {}).keys())
    # If snap config uses full names, keep them. Otherwise fall back to all.
    if not gnfr_list:
        gnfr_list = points_config.get("gnfr_full_names") or []

    cams_psrc = prepare_cams_emissions(nc_path, source_type="point", pollutants=list(POLLUTANT_NAMES.keys()), sectors=None)
    cams_asrc = prepare_cams_emissions(nc_path, source_type="area", pollutants=list(POLLUTANT_NAMES.keys()), sectors=None)

    cams_lons = cams_psrc.grid_lons
    cams_lats = cams_psrc.grid_lats
    nlat, nlon = len(cams_lats), len(cams_lons)

    xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs = float(cams_lons.min()), float(cams_lats.min()), float(cams_lons.max()), float(cams_lats.max())
    cams_transform_wgs = from_bounds(xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs, nlon, nlat)

    dom_wgs_bounds = domain_bounds_wgs84(domain_cfg)
    if logger.isEnabledFor(logging.DEBUG):
        if nlon > 1 and nlat > 1:
            res_lon = float(abs(cams_lons[1] - cams_lons[0]))
            res_lat = float(abs(cams_lats[1] - cams_lats[0]))
        else:
            res_lon = float("nan")
            res_lat = float("nan")
        logger.debug(
            "CAMS point grid (WGS84): "
            f"shape={nlat}x{nlon} "
            f"lon=[{float(cams_lons.min()):.3f},{float(cams_lons.max()):.3f}] "
            f"lat=[{float(cams_lats.min()):.3f},{float(cams_lats.max()):.3f}] "
            f"res≈({res_lon:.4f},{res_lat:.4f})"
        )
        logger.debug(f"Domain bounds WGS84: {dom_wgs_bounds}")

    logger.info("━━━ [2/6] Loading RI-URBANS point inventory ━━━━━━━━━━━━━")
    cols = points_config.get("columns", {})
    lon_col = cols.get("lon", "Lon")
    lat_col = cols.get("lat", "Lat")
    src_type_col = cols.get("source_type", "SourceType")
    gnfr_col = cols.get("gnfr_sector", "GNFR_Sector")

    ri_path = Path(points_config["riurbans_csv"])
    if not ri_path.is_absolute():
        ri_path = (Path(paths["data_root"]) / ri_path).resolve()
    ri = pd.read_csv(
        ri_path,
        sep=str(points_config.get("riurbans_sep", ";")),
        decimal=str(points_config.get("riurbans_decimal", ".")),
    )
    ri = ri[ri[src_type_col].astype(str) == str(points_config.get("riurbans_filter_source_type", "P"))].copy()
    ri = ri.rename(columns={lon_col: "lon", lat_col: "lat", gnfr_col: "gnfr"})

    xmin_d, ymin_d, xmax_d, ymax_d = dom_wgs_bounds
    ri = ri[(ri["lon"] >= xmin_d) & (ri["lon"] <= xmax_d) & (ri["lat"] >= ymin_d) & (ri["lat"] <= ymax_d)].copy()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"RI-URBANS after domain filter: rows={len(ri):,}")

    # Pollutant mapping RI-URBANS columns
    polmap: dict[str, str] = points_config.get("pollutant_map", {})
    for cams_key, ri_key in polmap.items():
        if ri_key in ri.columns:
            ri[ri_key] = pd.to_numeric(ri[ri_key], errors="coerce").fillna(0.0)

    ri = assign_cams_cell_indices(ri, cams_lons, cams_lats)
    # Keep only points that fell into grid
    ri = ri[(ri["lon_idx"] >= 0) & (ri["lon_idx"] < nlon) & (ri["lat_idx"] >= 0) & (ri["lat_idx"] < nlat)].copy()

    logger.info(f"  RI-URBANS points in domain: {len(ri):,}")
    if logger.isEnabledFor(logging.DEBUG) and len(ri):
        logger.debug(
            "RI-URBANS CAMS cell groups: "
            f"{ri.groupby(['gnfr', 'lat_idx', 'lon_idx']).ngroups:,}"
        )

    # Pre-project point coordinates once for output and rasterization
    ri_proj = project_points_to_domain(
        ri[["lon", "lat", "gnfr", "lon_idx", "lat_idx"] + [v for v in polmap.values() if v in ri.columns]].copy(),
        domain_crs,
    )

    logger.info("━━━ [3/6] Distributing CAMS point emissions to RI-URBANS ━")
    distributed_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    wgs84 = CRS.from_epsg(4326)

    # Group RI points by (gnfr code, lat_idx, lon_idx) for fast lookup
    ri_groups = ri_proj.groupby(["gnfr", "lat_idx", "lon_idx"], sort=False)

    # Precompute CAMS domain mask on the WGS84 grid (cell centers)
    lon_in_dom = (cams_lons >= xmin_d) & (cams_lons <= xmax_d)
    lat_in_dom = (cams_lats >= ymin_d) & (cams_lats <= ymax_d)

    for sector_full, sector_stack in cams_psrc.rasters.items():
        gnfr_code = str(sector_full).split("_", 1)[0]

        for cams_poll, pretty in POLLUTANT_NAMES.items():
            # CAMS stack uses pretty names
            if pretty not in sector_stack:
                continue
            arr = sector_stack[pretty]
            if arr is None:
                continue

            # Nonzero CAMS cells (point sources), filtered to domain extent
            nz = np.argwhere(arr > 0)
            if nz.size == 0:
                continue
            # Apply domain filter (center-in-bounds), matching R's subset() on lon/lat
            if lat_in_dom.any() and lon_in_dom.any():
                keep = lat_in_dom[nz[:, 0]] & lon_in_dom[nz[:, 1]]
                nz = nz[keep]
                if nz.size == 0:
                    continue

            cams_total = float(arr[arr > 0].sum())
            dist_total = 0.0
            unmatch_total = 0.0

            ri_poll_col = polmap.get(cams_poll)

            for lat_i, lon_i in nz:
                cams_emis = float(arr[lat_i, lon_i])
                key = (gnfr_code, int(lat_i), int(lon_i))
                if key not in ri_groups.indices:
                    unmatched_rows.append(
                        {"sector": gnfr_code, "pollutant": cams_poll, "lat_idx": int(lat_i), "lon_idx": int(lon_i), "emission": cams_emis}
                    )
                    unmatch_total += cams_emis
                    continue

                g = ri_groups.get_group(key)
                if ri_poll_col is None or ri_poll_col not in g.columns:
                    # No weights available -> unmatched
                    unmatched_rows.append(
                        {"sector": gnfr_code, "pollutant": cams_poll, "lat_idx": int(lat_i), "lon_idx": int(lon_i), "emission": cams_emis}
                    )
                    unmatch_total += cams_emis
                    continue

                weights_raw = g[ri_poll_col].to_numpy(float)
                tot_w = float(weights_raw.sum())
                if tot_w <= 0.0:
                    unmatched_rows.append(
                        {"sector": gnfr_code, "pollutant": cams_poll, "lat_idx": int(lat_i), "lon_idx": int(lon_i), "emission": cams_emis}
                    )
                    unmatch_total += cams_emis
                    continue

                w = weights_raw / tot_w
                distributed = cams_emis * w
                dist_total += float(distributed.sum())

                for idx_pt, emis_pt in enumerate(distributed):
                    distributed_rows.append(
                        {
                            "sector": gnfr_code,
                            "pollutant": cams_poll,
                            "xcor": float(g["xcor"].iloc[idx_pt]),
                            "ycor": float(g["ycor"].iloc[idx_pt]),
                            "lon": float(g["lon"].iloc[idx_pt]),
                            "lat": float(g["lat"].iloc[idx_pt]),
                            "emission": float(emis_pt),
                        }
                    )

            validation_rows.append(
                {
                    "sector": gnfr_code,
                    "pollutant": cams_poll,
                    "cams_total": cams_total,
                    "distributed": dist_total,
                    "unmatched": unmatch_total,
                    "total_distributed": dist_total + unmatch_total,
                    "difference": cams_total - (dist_total + unmatch_total),
                    "relative_diff_pct": (100.0 * (cams_total - (dist_total + unmatch_total)) / cams_total) if cams_total else 0.0,
                }
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[{gnfr_code}/{cams_poll}] "
                    f"cams_total={cams_total:.6g} "
                    f"distributed={dist_total:.6g} "
                    f"unmatched={unmatch_total:.6g} "
                    f"cells_nz={len(nz)}"
                )

    validation_df = pd.DataFrame(validation_rows)
    if intermediates_dir is not None:
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(intermediates_dir / "points_validation_balance.csv", index=False)

    logger.info(f"  distributed rows: {len(distributed_rows):,}")
    logger.info(f"  unmatched CAMS cells: {len(unmatched_rows):,}")
    if logger.isEnabledFor(logging.DEBUG) and len(unmatched_rows):
        tmp = pd.DataFrame(unmatched_rows)
        top = (
            tmp.groupby(["sector", "pollutant"])["emission"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        logger.debug("Top unmatched (sector,pollutant) totals:\n" + top.to_string())

    dist_df = pd.DataFrame(distributed_rows)

    logger.info("━━━ [4/6] Creating corrected area sources (unmatched→area) ━")
    unmatched_df = pd.DataFrame(unmatched_rows)
    corrected_area: dict[str, dict[str, np.ndarray]] = {}

    # Build the coarse grid (same as proxy coarse grid)
    cams_origin_arr, coarse_transform, ncol_c, nrow_c, coarse_bounds = coarse_grid_from_cams(
        cams_transform_wgs, nlat=nlat, nlon=nlon, domain_crs=domain_crs, domain_bounds=(xmin, ymin, xmax, ymax)
    )
    coarse_shape = (nrow_c, ncol_c)

    # Precompute unmatched points projected to domain CRS + coarse indices by lat/lon center
    if len(unmatched_df):
        # Convert CAMS cell center lon/lat to domain CRS for rasterization
        # First build centers for each unmatched record
        lons = cams_lons[unmatched_df["lon_idx"].to_numpy(int)]
        lats = cams_lats[unmatched_df["lat_idx"].to_numpy(int)]
        to_dom = Transformer.from_crs(wgs84, domain_crs, always_xy=True)
        xs, ys = to_dom.transform(lons.astype(float), lats.astype(float))
        unmatched_df = unmatched_df.copy()
        unmatched_df["xcor"] = xs
        unmatched_df["ycor"] = ys

    # Reproject each area-source sector/pollutant to coarse grid, add unmatched rasterized, then resample to domain
    for sector_full, sector_stack in cams_asrc.rasters.items():
        gnfr_code = str(sector_full).split("_", 1)[0]
        corrected_area[sector_full] = {}

        for cams_poll, pretty in POLLUTANT_NAMES.items():
            if pretty not in sector_stack:
                continue
            src_arr = sector_stack[pretty]
            if src_arr is None:
                continue

            coarse_arr = np.zeros(coarse_shape, dtype=np.float64)
            reproject(
                source=src_arr.astype(np.float64),
                destination=coarse_arr,
                src_transform=cams_transform_wgs,
                src_crs=wgs84,
                dst_transform=coarse_transform,
                dst_crs=domain_crs,
                resampling=WarpResampling.nearest,
            )

            # Add unmatched for this sector/pollutant
            add_sum = 0.0
            if len(unmatched_df):
                sel = (unmatched_df["sector"] == gnfr_code) & (unmatched_df["pollutant"] == cams_poll)
                if sel.any():
                    unmatched_layer = unmatched_df.loc[sel, ["xcor", "ycor", "emission"]]
                    raster_add = rasterize_unmatched_to_coarse(unmatched_layer, coarse_transform, coarse_shape)
                    add_sum = float(raster_add.sum())
                    coarse_arr = coarse_arr + raster_add

            # Resample coarse → domain (nearest)
            domain_arr = np.zeros(domain_shape, dtype=np.float64)
            reproject(
                source=coarse_arr,
                destination=domain_arr,
                src_transform=coarse_transform,
                src_crs=domain_crs,
                dst_transform=domain_transform,
                dst_crs=domain_crs,
                resampling=WarpResampling.nearest,
            )
            corrected_area[sector_full][pretty] = domain_arr

    logger.info("━━━ [5/6] Writing point-source UECT table ━━━━━━━━━━━━━━━")
    gnfr_to_snap: dict[str, int] = points_config.get("gnfr_to_snap", {})
    defaults = UectPointDefaults()

    if len(dist_df):
        # Build points table on CAMS pollutant keys, then map to UECT names
        wide = dist_df.pivot_table(
            index=["sector", "xcor", "ycor"],
            columns="pollutant",
            values="emission",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()

        wide["snap"] = wide["sector"].map(lambda s: int(gnfr_to_snap.get(str(s), -999)))
        wide["Hi"] = defaults.Hi
        wide["Vi"] = defaults.Vi
        wide["Ti"] = defaults.Ti
        wide["radi"] = defaults.radi

        # Ensure UECT columns exist in correct order (UECT expects pretty names)
        for cams_key, pretty in POLLUTANT_NAMES.items():
            if cams_key not in wide.columns:
                wide[cams_key] = 0.0

        out = wide[["snap", "xcor", "ycor", "Hi", "Vi", "Ti", "radi"]].copy()

        # Map to the UECT subset order used in area sources
        cams_order = ["nox", "nmvoc", "co", "sox", "nh3", "pm2_5", "pm10"]
        pretty_order = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]
        for ck, pk in zip(cams_order, pretty_order, strict=True):
            out[pk] = wide[ck].astype(float)

        # Optional CH4
        if "ch4" in wide.columns:
            out["CH4"] = wide["ch4"].astype(float)

        out["PN"] = np.nan
        out = out.replace({np.nan: -999})
        points_out = out
    else:
        points_out = pd.DataFrame(columns=["snap", "xcor", "ycor", "Hi", "Vi", "Ti", "radi"] + UECT_POLLUTANTS_ORDER + ["PN"])

    logger.info("━━━ [6/6] Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return points_out, corrected_area


def run_and_export(
    run_config_path: Path,
    points_config_path: Path,
    snap_config_path: Path,
) -> Path:
    run_config = load_run_config(run_config_path)
    configure_urbem_logging(debug=run_config.get("debug"))

    points_config = load_pointsources_config(points_config_path)
    snap_config = load_snap_mapping(snap_config_path)

    config_dir = run_config_path.parent
    paths = resolve_paths(run_config, config_dir)

    out_folder = paths["emission_output_folder"]
    out_folder.mkdir(parents=True, exist_ok=True)
    intermediates_dir = out_folder / "intermediates"

    points_df, corrected_area = run_point_sources(
        run_config, points_config, snap_config, paths, config_dir, intermediates_dir=intermediates_dir
    )

    # Write points CSV
    region = run_config.get("region", "Ioannina")
    stem = f"{region}_pointsources_CAMS-REG-AP_8.1"
    out_path = out_folder / f"{stem}_{len(points_df)}_psrc.csv"
    points_df.to_csv(out_path, index=False, na_rep="-999")

    # Write corrected area sources artifact
    corr_path = out_folder / f"{stem}_corrected_areasources.npz"
    flat = {f"{sector}::{poll}": arr for sector, stack in corrected_area.items() for poll, arr in stack.items()}
    np.savez_compressed(corr_path, **flat)

    return out_path

