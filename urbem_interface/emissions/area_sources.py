"""
Main area sources pipeline — replicate 2_UrbEm_areasources_CAMS8.1_Ioannina_v3.R

Orchestrates:
  1. prepare_cams_emissions
  2. project/crop/resample CAMS to domain
  3. proxy_cwd for each proxy
  4. proxy_distribution per GNFR
  5. UC factor for F1–F4 (optional)
  6. GNFR → SNAP mapping
  7. Select SNAPs, apply output remap
  8. Build UECT table, write CSV
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
    Resampling as WarpResampling,
)
from affine import Affine
from pyproj import Transformer

from urbem_interface.utils.config_loader import resolve_paths, load_run_config, load_proxies_config, load_snap_mapping
from urbem_interface.emissions.prepare_cams import prepare_cams_emissions, GNFR_FULL_NAMES, POLLUTANT_NAMES
from urbem_interface.emissions.proxy_preparation import proxy_cwd, proxy_distribution
from urbem_interface.utils.grid import _compute_proxy_coarse_grid, _load_and_warp_proxy, _warp_to_domain

from urbem_interface.logging_config import configure_urbem_logging, get_logger

logger = get_logger(__name__)

POLLUTANTS_OUT = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]

def _resolve_debug_pollutants(run_config: dict) -> tuple[list[str], list[str]]:
    """
    Return (pollutants_in, poll_names_out).

    - pollutants_in are CAMS NetCDF variable names (e.g. "nox", "co").
    - poll_names_out are pretty names used in outputs (e.g. "NOx", "CO").
    """
    req = run_config.get("debug_pollutants")
    if not req:
        pollutants_in = list(POLLUTANT_NAMES.keys())
        return pollutants_in, [POLLUTANT_NAMES[k] for k in pollutants_in]

    normalized: list[str] = []
    for p in req:
        s = str(p).strip()
        if not s:
            continue
        s_l = s.lower()
        if s_l in POLLUTANT_NAMES:
            normalized.append(s_l)
            continue
        # match by pretty name (case-insensitive), e.g. "NOx" -> "nox"
        key = next((k for k, v in POLLUTANT_NAMES.items() if v.lower() == s_l), None)
        if key:
            normalized.append(key)

    if not normalized:
        pollutants_in = list(POLLUTANT_NAMES.keys())
        return pollutants_in, [POLLUTANT_NAMES[k] for k in pollutants_in]

    # unique, keep order
    seen = set()
    pollutants_in = []
    for k in normalized:
        if k not in seen:
            pollutants_in.append(k)
            seen.add(k)
    return pollutants_in, [POLLUTANT_NAMES[k] for k in pollutants_in]





def _cache_key(domain_cfg: dict, year: int, nc_path: Path, paths: dict, proxies_config: dict) -> str:
    proxies_folder = paths["proxies_folder"]
    parts = [
        json.dumps(domain_cfg, sort_keys=True),
        str(year),
        str(nc_path),
        str(nc_path.stat().st_mtime) if nc_path.exists() else "",
    ]
    for name, fname in proxies_config.get("proxies", {}).items():
        p = proxies_folder / fname
        parts.append(f"{name}:{p.stat().st_mtime}" if p.exists() else f"{name}:")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _save_raster_stack_csv(
    sector_data: dict[str, np.ndarray],
    filepath: Path,
    nrow: int,
    ncol: int,
    poll_names: list[str] | None = None,
) -> None:
    """Save sector raster stack as CSV matching R format. Row/col 0-based, column-major."""
    if poll_names is None:
        poll_names = POLLUTANTS_OUT
    filepath.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in range(ncol):
        for r in range(nrow):
            row_dict = {"row": r, "col": c}
            for p in poll_names:
                arr = sector_data.get(p, np.zeros((nrow, ncol)))
                row_dict[p] = float(arr[r, c]) if not np.isnan(arr[r, c]) else 0.0
            rows.append(row_dict)
    pd.DataFrame(rows).to_csv(filepath, index=False)


def _save_raster_layer_csv(arr: np.ndarray, filepath: Path, nrow: int, ncol: int, colname: str = "value") -> None:
    """Save single raster layer as CSV matching R format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    flat = np.where(np.isnan(arr), 0.0, arr).astype(float).flatten(order="F")
    rows = []
    for c in range(ncol):
        for r in range(nrow):
            rows.append({"row": r, "col": c, colname: flat[c * nrow + r]})
    pd.DataFrame(rows).to_csv(filepath, index=False)

def _write_grid_metadata(
    intermediates_dir: Path,
    nrow: int, ncol: int,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    crs_str: str,
) -> None:
    """Write domain grid metadata CSV to intermediates directory."""
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    (intermediates_dir / "grid_metadata.csv").write_text(
        f"nrow,ncol,xmin,xmax,ymin,ymax,crs\n"
        f"{nrow},{ncol},{xmin},{xmax},{ymin},{ymax},{crs_str}\n"
    )
    logger.debug(f"  grid_metadata.csv written → {intermediates_dir}")


def run_area_sources(
    run_config: dict,
    proxies_config: dict,
    snap_config: dict,
    paths: dict,
    config_dir: Path,
    cache_dir: Path | None = None,
    intermediates_dir: Path | None = None,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> pd.DataFrame:
    """
    Run the full area sources pipeline. Returns the UECT DataFrame (before dropping zeros).
    """
    # ------------------------------------------------------------------ #
    # [1/8] Domain setup                                                   #
    # ------------------------------------------------------------------ #
    logger.info("━━━ [1/8] Domain config ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    domain_cfg   = run_config["domain"]
    year         = run_config["year"]
    pollutants_in, poll_names_out = _resolve_debug_pollutants(run_config)

    if run_config.get("debug_pollutants"):
        logger.debug(f"  debug_pollutants active → {poll_names_out}")

    nrow, ncol   = domain_cfg["nrow"], domain_cfg["ncol"]
    xmin, ymin   = domain_cfg["xmin"], domain_cfg["ymin"]
    xmax, ymax   = domain_cfg["xmax"], domain_cfg["ymax"]
    crs_str      = domain_cfg["crs"]
    domain_crs   = CRS.from_string(crs_str)
    domain_transform = from_bounds(xmin, ymin, xmax, ymax, ncol, nrow)
    domain_shape     = (nrow, ncol)
    domain_arr       = np.ones(domain_shape, dtype=np.float64)

    logger.info(
        f"  grid      : {nrow}×{ncol}  "
        f"x=[{xmin:.1f}, {xmax:.1f}]  y=[{ymin:.1f}, {ymax:.1f}]"
    )
    logger.info(f"  crs       : {crs_str}")
    logger.info(f"  year      : {year}")
    logger.info(f"  pollutants: {poll_names_out}")

    def emit(stage_id: str, data: dict | None = None):
        if progress_callback:
            progress_callback(stage_id, data or {})

    cams_folder    = paths["cams_folder"]
    proxies_folder = paths["proxies_folder"]

    nc_files = sorted(cams_folder.glob(f"*{year}.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF for year {year} in {cams_folder}")
    nc_path = nc_files[0]
    logger.info(f"  NetCDF    : {nc_path.name}")

    gnfr_list = list(proxies_config["gnfr_to_proxy"].keys())
    logger.info(f"  GNFR sectors ({len(gnfr_list)}): {gnfr_list}")

    # ------------------------------------------------------------------ #
    # Cache check                                                          #
    # ------------------------------------------------------------------ #
    cache_key = _cache_key(domain_cfg, year, nc_path, paths, proxies_config) if cache_dir else None
    cache_ok  = False
    if cache_dir:
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                cache_ok = manifest.get("key") == cache_key
            except (json.JSONDecodeError, OSError):
                cache_ok = False

    # ------------------------------------------------------------------ #
    # [2/8] + [3/8]  CAMS read & project  (or cache restore)             #
    # ------------------------------------------------------------------ #
    if cache_ok:
        emit("cams")
        logger.info(f"━━━ [2/8] Cache hit — restoring  (key={cache_key}) ━━━━━━━")
        gnfr_raster: dict[str, dict[str, np.ndarray]] = {}
        with np.load(cache_dir / "gnfr_raster.npz") as z:
            for k in z.files:
                sector, poll = k.split("::", 1)
                gnfr_raster.setdefault(sector, {})[poll] = z[k].copy()

        emit("proxies")
        normalized_proxies: dict[str, np.ndarray] = {}
        with np.load(cache_dir / "proxies.npz") as z:
            for name in z.files:
                normalized_proxies[name] = z[name].copy()

        logger.info(
            f"  restored  : {len(gnfr_raster)} sectors, "
            f"{len(normalized_proxies)} proxies"
        )
        logger.info("━━━ [3/8] Skipped (cache) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        if logger.isEnabledFor(logging.DEBUG):
            for sn, sd in gnfr_raster.items():
                nox = float(np.nansum(sd.get("NOx", np.zeros(domain_shape))))
                logger.debug(f"  cache [{sn}]  NOx={nox:.1f}")

        if intermediates_dir:
            _write_grid_metadata(intermediates_dir, nrow, ncol, xmin, xmax, ymin, ymax, crs_str)
            for sn, sd in gnfr_raster.items():
                _save_raster_stack_csv(
                    sd,
                    intermediates_dir / "step1_cams_warped" / f"{sn.replace('_', '-')}.csv",
                    nrow, ncol, poll_names=poll_names_out,
                )
            for pn, arr in normalized_proxies.items():
                _save_raster_layer_csv(
                    arr,
                    intermediates_dir / "step2_proxies" / f"{pn}.csv",
                    nrow, ncol, colname="weight",
                )

    else:
        emit("cams")
        # -------------------------------------------------------------- #
        # [2/8] Read CAMS NetCDF                                          #
        # -------------------------------------------------------------- #
        logger.info(f"━━━ [2/8] Reading CAMS NetCDF ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        cams_result = prepare_cams_emissions(
            nc_path, source_type="area",
            pollutants=pollutants_in, sectors=gnfr_list,
        )

        wgs84        = CRS.from_epsg(4326)
        cams_lon     = cams_result.grid_lons
        cams_lat     = cams_result.grid_lats
        nlat, nlon   = len(cams_lat), len(cams_lon)
        xmin_wgs     = float(cams_lon.min());  xmax_wgs = float(cams_lon.max())
        ymin_wgs     = float(cams_lat.min());  ymax_wgs = float(cams_lat.max())
        cams_transform_wgs = from_bounds(xmin_wgs, ymin_wgs, xmax_wgs, ymax_wgs, nlon, nlat)

        logger.info(
            f"  CAMS grid : {nlat}×{nlon}  "
            f"lon=[{xmin_wgs:.3f}, {xmax_wgs:.3f}]  "
            f"lat=[{ymin_wgs:.3f}, {ymax_wgs:.3f}]"
        )

        # -------------------------------------------------------------- #
        # [3/8] Project CAMS → domain                                     #
        # -------------------------------------------------------------- #
        logger.info("━━━ [3/8] Projecting CAMS to domain ━━━━━━━━━━━━━━━━━━━━━")
        gnfr_raster = {}
        for sector_name in gnfr_list:
            if sector_name not in cams_result.rasters:
                logger.debug(f"  [{sector_name}] not in CAMS rasters — skipped")
                continue
            sector_data  = cams_result.rasters[sector_name]
            sector_stack = {}
            for poll, arr in sector_data.items():
                if logger.isEnabledFor(logging.DEBUG):
                    nz  = int(np.count_nonzero(arr))
                    nan = int(np.sum(np.isnan(arr)))
                    logger.debug(
                        f"  [{sector_name}/{poll}]  "
                        f"shape={arr.shape}  nonzero={nz}  nan={nan}  "
                        f"sum={float(np.nansum(arr)):.3g}  "
                        f"min={float(np.nanmin(arr)):.3g}  max={float(np.nanmax(arr)):.3g}"
                    )
                sector_stack[poll] = _warp_to_domain(
                    arr, cams_transform_wgs, wgs84,
                    domain_transform, domain_shape, domain_crs,
                    domain_bounds=(xmin, ymin, xmax, ymax),
                    resampling="nearest",
                    label=f"{sector_name}-{poll}",
                )
            gnfr_raster[sector_name] = sector_stack
            if logger.isEnabledFor(logging.DEBUG):
                nox = float(np.nansum(sector_stack.get("NOx", np.zeros(domain_shape))))
                logger.debug(f"  [{sector_name}] warped — NOx={nox:.1f}")

        logger.info(f"  projected {len(gnfr_raster)} sectors to domain grid")

        if intermediates_dir:
            _write_grid_metadata(intermediates_dir, nrow, ncol, xmin, xmax, ymin, ymax, crs_str)
            for sn, sd in gnfr_raster.items():
                _save_raster_stack_csv(
                    sd,
                intermediates_dir / "step1_cams_warped" / f"{sn.replace('_', '-')}.csv",
                nrow, ncol, poll_names=poll_names_out,
            )

        # -------------------------------------------------------------- #
        # [4/8] Coarse proxy grid                                          #
        # -------------------------------------------------------------- #
        logger.info("━━━ [4/8] Building coarse proxy grid ━━━━━━━━━━━━━━━━━━━━")
        cams_origin_arr, cams_origin_transform, nlon_c, nlat_c, coarse_bounds = \
            _compute_proxy_coarse_grid(
                src_shape=(nlat, nlon),
                src_transform=cams_transform_wgs,
                src_crs=wgs84,
                dst_crs=domain_crs,
                domain_bounds=(xmin, ymin, xmax, ymax),
                res_x=7000.0, res_y=5550.0,
            )
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = coarse_bounds
        logger.info(
            f"  coarse grid: {nlat_c}×{nlon_c} cells  "
            f"bounds=({crop_xmin:.1f}, {crop_ymin:.1f}, {crop_xmax:.1f}, {crop_ymax:.1f})"
        )

        if intermediates_dir:
            coarse_dir = intermediates_dir / "step2_coarse_grid"
            coarse_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "nrow": nlat_c, "ncol": nlon_c,
                "xmin": crop_xmin, "ymin": crop_ymin,
                "xmax": crop_xmax, "ymax": crop_ymax,
                "res_x": float(cams_origin_transform.a),
                "res_y": float(abs(cams_origin_transform.e)),
                "crs": crs_str,
            }]).to_csv(coarse_dir / "cams_origin_metadata.csv", index=False)

            rows_grid = [
                {"row": r, "col": c, "cell_id": int(cams_origin_arr[r, c])}
                for c in range(nlon_c) for r in range(nlat_c)
            ]
            pd.DataFrame(rows_grid).to_csv(coarse_dir / "cams_origin_grid.csv", index=False)

        # -------------------------------------------------------------- #
        # [5/8] Load & normalize proxies                                   #
        # -------------------------------------------------------------- #
        emit("proxies")
        logger.info("━━━ [5/8] Normalizing proxies ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        proxy_names        = list(proxies_config["proxies"].keys())
        normalized_proxies = {}

        for proxy_name in proxy_names:
            proxy_file = proxies_config["proxies"][proxy_name]
            proxy_path = proxies_folder / proxy_file
            if not proxy_path.exists():
                logger.warning(f"  [{proxy_name}] SKIP — file not found: {proxy_path}")
                continue

            proxy_resampling = WarpResampling.nearest if proxy_name == "offroad" \
                               else WarpResampling.bilinear
            proxy_arr = _load_and_warp_proxy(
                proxy_path, domain_transform, domain_shape, domain_crs,
                resampling=proxy_resampling,
            )
            norm = proxy_cwd(
                cams_origin_arr, cams_origin_transform, domain_crs,
                domain_arr,      domain_transform,      domain_crs,
                proxy_arr,       domain_transform,      domain_crs,
                sparse_threshold=0.05,
                max_weight_per_cell=0.5,
            )
            normalized_proxies[proxy_name] = norm

            if logger.isEnabledFor(logging.DEBUG):
                nonzero = int(np.count_nonzero(norm))
                logger.debug(
                    f"  [{proxy_name}]  "
                    f"nonzero={nonzero}/{norm.size}  "
                    f"sum={float(norm.sum()):.4g}  "
                    f"max={float(norm.max()):.4g}"
                )
            logger.info(f"  [{proxy_name}] normalized ✓")

            if intermediates_dir:
                _save_raster_layer_csv(
                    norm,
                    intermediates_dir / "step2_proxies" / f"{proxy_name}.csv",
                    nrow, ncol, colname="weight",
                )

        logger.info(
            f"  {len(normalized_proxies)}/{len(proxy_names)} proxies normalized"
        )

        # -------------------------------------------------------------- #
        # Cache save                                                       #
        # -------------------------------------------------------------- #
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_dir / "gnfr_raster.npz",
                **{f"{s}::{p}": arr
                   for s, data in gnfr_raster.items()
                   for p, arr in data.items()},
            )
            np.savez_compressed(cache_dir / "proxies.npz", **normalized_proxies)
            (cache_dir / "manifest.json").write_text(json.dumps({"key": cache_key}))
            logger.info(f"  cache saved → {cache_dir}")

    # ------------------------------------------------------------------ #
    # [6/8] UC factor                                                      #
    # ------------------------------------------------------------------ #
    logger.info("━━━ [6/8] UC factor ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    uc_gnfr   = set(proxies_config.get("uc_apply_to_gnfr", []))
    uc_factor = float(proxies_config.get("uc_factor", 3))
    ghsl_path = proxies_folder / proxies_config.get("ghsl_urbancentre", "ghs_europe_iso3.tif")

    uc_mult = None
    if uc_gnfr and ghsl_path.exists():
        ghsl_arr = _load_and_warp_proxy(
            ghsl_path, domain_transform, domain_shape, domain_crs,
            resampling=WarpResampling.nearest,
        )
        uc_mult = np.where(ghsl_arr > 0, uc_factor, 1.0)
        urban_frac = float((ghsl_arr > 0).mean())
        logger.info(
            f"  GHSL loaded  urban_frac={urban_frac:.2%}  "
            f"factor={uc_factor}  applies to: {sorted(uc_gnfr)}"
        )
    else:
        logger.info(
            "  skipped — "
            + ("no GHSL file" if not ghsl_path.exists() else "uc_apply_to_gnfr is empty")
        )

    # ------------------------------------------------------------------ #
    # [7/8] Proxy distribution + GNFR→SNAP mapping                        #
    # ------------------------------------------------------------------ #
    logger.info("━━━ [7/8] Proxy distribution & GNFR→SNAP mapping ━━━━━━━━")
    _zero = np.zeros(domain_shape)   # reusable sentinel — avoids repeated allocation

    for sector_name, sector_data in gnfr_raster.items():
        proxy_name = proxies_config["gnfr_to_proxy"].get(sector_name)
        if not proxy_name or proxy_name not in normalized_proxies:
            logger.debug(f"  [{sector_name}] no proxy (proxy={proxy_name}) — skipped")
            continue

        out_dist = proxy_distribution(sector_data, normalized_proxies[proxy_name], "coarse_cells_proxy")

        if uc_mult is not None and sector_name in uc_gnfr:
            for poll in out_dist:
                out_dist[poll] = out_dist[poll] * uc_mult
            logger.debug(f"  [{sector_name}] UC factor applied")

        gnfr_raster[sector_name] = out_dist

        if logger.isEnabledFor(logging.DEBUG):
            nox = float(np.nansum(out_dist.get("NOx", _zero)))
            logger.debug(f"  [{sector_name}] after proxy  NOx={nox:.1f}")

        if intermediates_dir:
            _save_raster_stack_csv(
                out_dist,
                intermediates_dir / "step3_after_proxy" / f"{sector_name.replace('_', '-')}.csv",
                nrow, ncol, poll_names=poll_names_out,
            )

    emit("downscale")
    gnfr_to_snap = snap_config["gnfr_to_snap"]
    snap_rasters: dict[int, dict[str, np.ndarray]] = {}

    for key, mapping in gnfr_to_snap.items():
        snap  = mapping["snap"]
        split = float(mapping.get("split", 1.0))
        gnfr_base = key.replace("_snap4", "").replace("_snap3", "")
        if gnfr_base not in gnfr_raster:
            continue
        snap_rasters.setdefault(snap, {p: np.zeros(domain_shape) for p in poll_names_out})
        for p, arr in gnfr_raster[gnfr_base].items():
            if p in snap_rasters[snap]:
                snap_rasters[snap][p] += arr * split

    if logger.isEnabledFor(logging.DEBUG):
        for snap_id, sr in sorted(snap_rasters.items()):
            nox = float(np.nansum(sr.get("NOx", _zero)))
            logger.debug(f"  SNAP{snap_id:>2}  NOx={nox:.1f}")

    logger.info(f"  {len(snap_rasters)} SNAP sectors assembled")

    if intermediates_dir:
        for snap_id, sr in sorted(snap_rasters.items()):
            _save_raster_stack_csv(
                sr,
                intermediates_dir / "step4_snap" / f"snap{snap_id}.csv",
                nrow, ncol, poll_names=poll_names_out,
            )

    # ------------------------------------------------------------------ #
    # [8/8] Build UECT table (vectorized)                                  #
    # ------------------------------------------------------------------ #
    logger.info("━━━ [8/8] Building UECT table ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    snap_export = snap_config["snap_sectors_export"]
    snap_remap  = snap_config.get("snap_output_remap", {})
    snap_height = snap_config.get("snap_emission_height", {})

    cell_w = (xmax - xmin) / ncol
    cell_h = (ymax - ymin) / nrow

    # Pre-build coordinate arrays once (vectorized, not per-cell)
    col_idx = np.arange(ncol, dtype=np.float64)
    row_idx = np.arange(nrow, dtype=np.float64)

    x_centers = xmin + (col_idx + 0.5) * cell_w   # (ncol,)
    y_centers  = ymax - (row_idx + 0.5) * cell_h   # (nrow,)

    # full-grid meshes — shape (nrow*ncol,)
    x_c = np.tile(x_centers, nrow)                  # col varies fastest
    y_c = np.repeat(y_centers, ncol)

    sw_x = x_c - cell_w / 2;  ne_x = x_c + cell_w / 2
    sw_y = y_c - cell_h / 2;  ne_y = y_c + cell_h / 2

    blocks = []
    for snap_in in snap_export:
        if snap_in not in snap_rasters:
            continue
        snap_out = int(snap_remap.get(str(snap_in), snap_in))
        z        = int(snap_height.get(str(snap_in), 10))
        data     = snap_rasters[snap_in]

        # Stack all pollutants → (n_polls, nrow*ncol), NaN→0
        poll_matrix = np.stack(
            [np.nan_to_num(data.get(p, _zero), nan=0.0).ravel() for p in poll_names_out],
            axis=1,
        )  # shape (nrow*ncol, n_polls)

        n = nrow * ncol
        snap_col = np.full(n, snap_out, dtype=np.int32)
        z_col    = np.full(n, z,        dtype=np.int32)

        block = np.column_stack([snap_col, sw_x, sw_y, z_col, ne_x, ne_y, z_col, poll_matrix])
        blocks.append(block)

    if blocks:
        raw = np.vstack(blocks)
        df  = pd.DataFrame(
            raw,
            columns=["snap", "xcor_sw", "ycor_sw", "zcor_sw",
                     "xcor_ne", "ycor_ne", "zcor_ne"] + poll_names_out,
        )
        df["snap"]    = df["snap"].astype(int)
        df["zcor_sw"] = df["zcor_sw"].astype(int)
        df["zcor_ne"] = df["zcor_ne"].astype(int)
    else:
        df = pd.DataFrame(
            columns=["snap", "xcor_sw", "ycor_sw", "zcor_sw",
                     "xcor_ne", "ycor_ne", "zcor_ne"] + poll_names_out
        )

    df.replace(np.nan, -999, inplace=True)

    logger.info(
        f"  rows={len(df):,}  cols={len(df.columns)}  "
        f"snaps={sorted(df['snap'].unique().tolist()) if len(df) else []}"
    )
    logger.info("━━━ Pipeline complete ✓ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return df


def run_and_export(
    run_config_path: Path,
    proxies_config_path: Path,
    snap_config_path: Path,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> Path:
    """Load configs, run pipeline, drop zero rows, write CSV. Returns output path."""
    run_config = load_run_config(run_config_path)
    configure_urbem_logging(debug=run_config.get("debug"))

    snap_config = load_snap_mapping(snap_config_path)
    config_dir = run_config_path.parent
    paths = resolve_paths(run_config, config_dir)
    proxies_config = load_proxies_config(
        proxies_config_path, proxies_folder=paths["proxies_folder"]
    )

    out_folder = paths["emission_output_folder"]
    out_folder.mkdir(parents=True, exist_ok=True)
    cache_dir = out_folder / ".urbem_cache"
    intermediates_dir = out_folder / "intermediates"

    try:
        df = run_area_sources(
            run_config, proxies_config, snap_config, paths, config_dir,
            cache_dir=cache_dir,
            intermediates_dir=intermediates_dir,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    _, poll_cols = _resolve_debug_pollutants(run_config)
    non_zero = df[df[poll_cols].sum(axis=1) > 0]
    logger.info(f"Non-zero cells: {len(non_zero)}")
    out_folder.mkdir(parents=True, exist_ok=True)
    region = run_config.get("region", "Ioannina")
    stem = f"{region}_areasources_CAMS-REG-AP_8.1"
    out_path = out_folder / f"{stem}_{len(non_zero)}_asrc.csv"
    non_zero.to_csv(out_path, index=False, na_rep="-999")
    return out_path
