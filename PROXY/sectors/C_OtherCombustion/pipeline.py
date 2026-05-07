"""
GNFR C other-combustion **pipeline** orchestration (CAMS allocation + GeoTIFF outputs).

**Role**: single entry ``run`` / ``run_other_combustion_weight_build`` used by
``C_OtherCombustion`` builder; wires validation, Eurostat factors, ``M`` and ``X``,
and CAMS cell raster accumulation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from PROXY.core.cams.domain import country_index_1based as _country_index_1based
from PROXY.core.cams.domain import domain_mask_wgs84 as _build_domain_mask
from PROXY.core.cams.gnfr import gnfr_code_to_index as _gnfr_to_index
from PROXY.core.dataloaders import resolve_path
from PROXY.core.io import write_geotiff

from .constants import MODEL_CLASSES
from .exceptions import ConfigurationError
from .m_builder.assemble import build_m_matrix
from .m_builder.emep_ef import load_emep
from .m_builder.enduse_factors import compute_end_use_factors, log_enduse_tables
from .m_builder.gains_activity import index_gains_files
from .m_builder.mapping_io import load_gains_mapping
from .allocator import accumulate_emissions_and_weights_for_cell
from .validation import validate_pipeline_config
from .x_builder.stack import load_and_build_fields
from ._log import LOG


def _dataframe_to_log_block(df: Any) -> str:
    try:
        return str(df.to_string())
    except Exception:
        return repr(df)


def _log_M_pol_times_class(iso3: str, M: np.ndarray, pollutant_outputs: list[str]) -> None:
    try:
        import pandas as pd

        df = pd.DataFrame(
            np.asarray(M, dtype=np.float64),
            index=[str(p) for p in pollutant_outputs],
            columns=[str(c) for c in MODEL_CLASSES],
        )
        LOG.info(
            "[other_combustion] GAINS×EMEP matrix M (iso3=%s, rows=pollutants, cols=model_class):\n%s",
            iso3,
            _dataframe_to_log_block(df),
        )
    except ImportError:
        hdr = "pollutant," + ",".join(MODEL_CLASSES)
        lines = [f"[other_combustion] M matrix (iso3={iso3})", hdr]
        for ri, pol in enumerate(pollutant_outputs):
            row = [str(pol)] + [f"{float(M[ri, ci]):.6g}" for ci in range(len(MODEL_CLASSES))]
            lines.append(",".join(row))
        LOG.info("\n".join(lines))


def _log_X_band_summaries(X: np.ndarray) -> None:
    h, w, k = X.shape
    lines = [f"[other_combustion] proxy stack X: shape=({h}, {w}, {k}) bands={list(MODEL_CLASSES)}"]
    flat_n = max(h * w, 1)
    for ki, name in enumerate(MODEL_CLASSES):
        band = np.asarray(X[:, :, ki], dtype=np.float64).ravel()
        m = np.isfinite(band)
        n_fin = int(np.count_nonzero(m))
        frac = 100.0 * n_fin / flat_n
        if n_fin == 0:
            lines.append(f"  {name}: finite=0.0% (empty)")
            continue
        sub = band[m]
        lines.append(
            f"  {name}: finite={frac:.1f}% min={float(np.min(sub)):.6g} "
            f"p50={float(np.median(sub)):.6g} max={float(np.max(sub)):.6g}"
        )
    LOG.info("\n".join(lines))


def _decode_country_ids(ds: xr.Dataset) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _iso3_for_source(i: int, country_idx: np.ndarray, codes: list[str]) -> str:
    ix = int(country_idx[i]) - 1
    if 0 <= ix < len(codes):
        return str(codes[ix]).strip().upper()
    return "UNK"


def _cell_bounds_overlap_domain_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    bbox: tuple[float, float, float, float],
) -> bool:
    bw, bs, be, bn = bbox
    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south
    return not (east < bw or west > be or north < bs or south > bn)


def run(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    output_weights_path: Path,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    paths0 = cfg.get("paths") or {}
    xlsx = paths0.get("eurostat_xlsx")
    if xlsx is not None and str(xlsx).strip() and str(xlsx).strip().lower() not in {"null", "none"}:
        raise ConfigurationError(
            "paths.eurostat_xlsx is no longer supported. Remove it from paths.yaml / merged config. "
            "Household end-uses and commercial split are loaded from the Eurostat API with on-disk cache "
            "(see PROXY/sectors/C_OtherCombustion/eurostat_api.py and eurostat.enabled in sector YAML)."
        )

    paths = cfg["paths"]
    nc = resolve_path(repo_root, paths["cams_nc"])
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")

    iso_for_enduse = str(cfg["country"]["cams_iso3"]).strip().upper()
    LOG.info("[other_combustion] CAMS NetCDF=%s", nc)
    LOG.info("[other_combustion] focus country cams_iso3=%s", iso_for_enduse)

    emep_path = resolve_path(repo_root, paths["emep_ef"])
    emep = load_emep(emep_path)
    mapping_path = resolve_path(repo_root, paths["gains_mapping"])
    rules, emep_fuel_hints = load_gains_mapping(mapping_path)
    sidecar_path = resolve_path(repo_root, paths["eurostat_end_use_json"])

    pollutant_specs: list[dict[str, Any]] = []
    for p in cfg.get("pollutants") or []:
        pollutant_specs.append(dict(p))
    if not pollutant_specs:
        raise ValueError("config pollutants list is empty")

    pollutant_outputs = [str(p["output"]) for p in pollutant_specs]
    validate_pipeline_config(
        repo_root=repo_root,
        cfg=cfg,
        mapping_path=mapping_path,
        emep_path=emep_path,
        sidecar_path=sidecar_path,
        pollutant_outputs=pollutant_outputs,
    )

    LOG.info("[other_combustion] EMEP factors sidecar=%s", emep_path)
    LOG.info("[other_combustion] GAINS mapping sidecar=%s", mapping_path)

    gains_dir = resolve_path(repo_root, paths["gains_dir"])
    overrides = (cfg.get("gains") or {}).get("iso3_file_overrides") or {}
    gains_index = index_gains_files(gains_dir, overrides, repo_root)
    year_col = str((cfg.get("gains") or {}).get("year_column", "2020"))
    LOG.info(
        "[other_combustion] GAINS dir=%s indexed_iso3_count=%d year_column=%r",
        gains_dir,
        len(gains_index),
        year_col,
    )

    gains_focus = gains_index.get(iso_for_enduse)
    factors = compute_end_use_factors(
        repo_root=repo_root,
        cfg=cfg,
        iso3=iso_for_enduse,
        gains_path=gains_focus,
        year_col=year_col,
        rules=rules,
    )
    log_enduse_tables(iso_for_enduse, factors)

    co2_mode = str((cfg.get("co2") or {}).get("mode", "sum_ff_bf"))
    LOG.info("[other_combustion] pollutants=%s co2_mode=%r", pollutant_outputs, co2_mode)

    run_cfg = cfg.get("run") or {}
    write_weights = bool(run_cfg.get("write_weights_geotiff", True))
    write_emissions = bool(run_cfg.get("write_emissions_geotiffs", False))
    write_per_pollutant_w = bool(run_cfg.get("write_weight_geotiffs", False))
    need_weights = write_weights or write_per_pollutant_w

    if show_progress is None:
        show_progress = bool(run_cfg.get("show_progress", True))
    else:
        show_progress = bool(show_progress)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    fields = load_and_build_fields(repo_root, cfg, ref)
    X = fields["X"]
    H, W, K = X.shape
    assert K == len(MODEL_CLASSES)
    _log_X_band_summaries(X)

    t_ref = ref["transform"]
    crs_s = str(ref["crs"])
    bbox_wgs = tuple(float(x) for x in ref["domain_bbox_wgs84"])
    LOG.info(
        "[other_combustion] reference grid height=%d width=%d crs=%s domain_bbox_wgs84=%s",
        H,
        W,
        crs_s,
        bbox_wgs,
    )

    out_dir = cfg.get("output_dir")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc)
    uniform_fb = 0
    processed_cells = 0
    try:
        iso_filter = str(cfg["country"]["cams_iso3"]).strip().upper()
        country_1b = _country_index_1based(ds, iso_filter)
        emis_idx = _gnfr_to_index(str((cfg.get("cams") or {}).get("gnfr", "C")))
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        bbox_cfg = (cfg.get("cams") or {}).get("domain_bbox_wgs84")
        bbox_use = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
        dom = _build_domain_mask(lon, lat, ci, country_1b, bbox_use)
        source_types = (cfg.get("cams") or {}).get("source_types") or ["area"]
        mask = dom & (emis == emis_idx)
        if "area" in source_types and "point" not in source_types:
            mask = mask & (st == 1)
        elif "point" in source_types and "area" not in source_types:
            mask = mask & (st == 2)
        elif "area" in source_types and "point" in source_types:
            mask = mask & ((st == 1) | (st == 2))

        codes = _decode_country_ids(ds)
        idx_cells = np.flatnonzero(mask)
        matrix_by_iso3: dict[str, np.ndarray] = {}

        LOG.info(
            "[other_combustion] CAMS mask — %d of %d sources selected (GNFR index=%s, source_types=%s)",
            int(idx_cells.size),
            int(mask.size),
            emis_idx,
            source_types,
        )

        acc = np.zeros((len(pollutant_specs), H, W), dtype=np.float64)
        weights_acc = (
            np.zeros((len(pollutant_specs), H, W), dtype=np.float64) if need_weights else None
        )

        lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
        lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
        if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
            lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
            lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
        lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
        lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)

        for i in idx_cells:
            iso = _iso3_for_source(int(i), ci, codes)
            if iso in matrix_by_iso3 or iso == "UNK":
                continue
            gp = gains_index.get(iso)
            matrix_by_iso3[iso] = build_m_matrix(
                gp,
                year_col,
                rules,
                factors,
                emep,
                pollutant_outputs,
                emep_fuel_hints=emep_fuel_hints,
            )

        for iso3_key, M in sorted(matrix_by_iso3.items()):
            if iso3_key == iso_for_enduse:
                _log_M_pol_times_class(iso3_key, M, pollutant_outputs)
            else:
                LOG.debug(
                    "[other_combustion] M matrix for iso3=%s (summary col_sum_L1=%s)",
                    iso3_key,
                    float(np.sum(np.abs(M))),
                )

        iso_in_cells = {_iso3_for_source(int(i), ci, codes) for i in idx_cells}
        missing_gains = sorted(iso_in_cells - set(matrix_by_iso3.keys()) - {"UNK"})
        if missing_gains:
            LOG.warning(
                "[other_combustion] CAMS cells include ISO3 with no GAINS file (M will be zeros): %s",
                missing_gains,
            )

        cell_iter = idx_cells
        if show_progress and tqdm is not None:
            cell_iter = tqdm(
                idx_cells,
                desc="CAMS cells (GNFR C other combustion)",
                unit="cell",
                total=int(idx_cells.size),
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
                mininterval=0.5,
            )

        for i in cell_iter:
            li, ji = int(lon_ii[i]), int(lat_ii[i])
            west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
            south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west
            if not _cell_bounds_overlap_domain_bbox(west, south, east, north, bbox_wgs):
                continue

            poly4326 = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
            g3035 = poly4326.to_crs(crs_s)
            geom = g3035.geometry.iloc[0]
            minx, miny, maxx, maxy = geom.bounds
            try:
                win = from_bounds(minx, miny, maxx, maxy, transform=t_ref).intersection(Window(0, 0, W, H))
            except WindowError:
                continue
            win = win.round_lengths().round_offsets()
            if win.width < 1 or win.height < 1:
                continue
            r0, c0 = int(win.row_off), int(win.col_off)
            h_win, w_win = int(win.height), int(win.width)
            Xw = X[r0 : r0 + h_win, c0 : c0 + w_win, :].reshape(-1, K).astype(np.float64)
            iso = _iso3_for_source(int(i), ci, codes)
            M = matrix_by_iso3.get(iso)
            if M is None:
                M = np.zeros((len(pollutant_outputs), K), dtype=np.float64)
            U = Xw @ M.T
            rr, cc = np.meshgrid(
                np.arange(r0, r0 + h_win, dtype=np.int32),
                np.arange(c0, c0 + w_win, dtype=np.int32),
                indexing="ij",
            )
            flat_r = rr.ravel()
            flat_c = cc.ravel()
            cell_used_fb = accumulate_emissions_and_weights_for_cell(
                U=U,
                pollutant_specs=pollutant_specs,
                ds=ds,
                cell_index=int(i),
                co2_mode=co2_mode,
                flat_r=flat_r,
                flat_c=flat_c,
                acc=acc,
                weights_acc=weights_acc,
            )
            processed_cells += 1
            if cell_used_fb:
                uniform_fb += 1
    finally:
        ds.close()

    if processed_cells > 0:
        pct = 100.0 * uniform_fb / max(processed_cells, 1)
        LOG.info(
            "[other_combustion] %s: %d/%d CAMS overlap windows used uniform fallback (%.2f%%)",
            iso_for_enduse,
            uniform_fb,
            processed_cells,
            pct,
        )

    profile_single = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": crs_s,
        "transform": t_ref,
        "compress": "lzw",
    }
    if write_emissions and out_dir is not None:
        for pi, spec in enumerate(pollutant_specs):
            out_tif = out_dir / f"emissions_{spec['output']}.tif"
            with rasterio.open(out_tif, "w", **profile_single) as dst:
                dst.write(acc[pi].astype(np.float32), 1)
                dst.set_band_description(1, f"kg_yr_{spec['output']}_GNFR_C_allocated")
        LOG.info("[other_combustion] wrote emissions GeoTIFF(s) under %s", out_dir)

    if write_weights and weights_acc is not None:
        n_b = len(pollutant_specs)
        stack = np.stack([weights_acc[i] for i in range(n_b)], axis=0).astype(np.float32)
        band_desc = [f"weight_share_gnfr_c_{str(spec['output']).lower()}" for spec in pollutant_specs]
        write_geotiff(
            path=output_weights_path,
            array=stack,
            crs=crs_s,
            transform=t_ref,
            band_descriptions=band_desc,
        )
        LOG.info("[other_combustion] wrote multiband proxy weights -> %s", output_weights_path)

    if write_per_pollutant_w and weights_acc is not None and out_dir is not None:
        for pi, spec in enumerate(pollutant_specs):
            wtif = out_dir / f"weights_{spec['output']}.tif"
            with rasterio.open(wtif, "w", **profile_single) as dst:
                dst.write(weights_acc[pi].astype(np.float32), 1)
        LOG.info("[other_combustion] wrote per-pollutant weight GeoTIFF(s) under %s", out_dir)

    return {
        "output_path": str(output_weights_path),
        "height": H,
        "width": W,
        "crs": crs_s,
        "domain_bbox_wgs84": list(bbox_wgs),
        "pollutant_bands": len(pollutant_specs),
        "gains_files_indexed": len(gains_index),
    }


def run_other_combustion_weight_build(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    output_weights_path: Path,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Public entry (backward compatible module path)."""
    return run(
        repo_root=repo_root,
        cfg=cfg,
        ref=ref,
        output_weights_path=output_weights_path,
        show_progress=show_progress,
    )


def run_downscale(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    output_weights_path: Path,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Deprecated alias."""
    return run_other_combustion_weight_build(
        repo_root=repo_root,
        cfg=cfg,
        ref=ref,
        output_weights_path=output_weights_path,
        show_progress=show_progress,
    )
