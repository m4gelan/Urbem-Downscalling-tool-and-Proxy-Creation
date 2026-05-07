from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling

from PROXY.core.cams.grid import build_cam_cell_id
from PROXY.core.ceip import DEFAULT_GNFR_GROUP_ORDER, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import resolve_path
from PROXY.core.diagnostics import RasterStatsLogger, max_valid_cam_cell_id
from PROXY.core.logging_tables import log_wide_group_alpha_table
from PROXY.core.osm_corine_proxy import build_all_group_pg, build_p_pop
from PROXY.sectors.D_Fugitive.fugitive_proxy import build_fugitive_group_pg
from PROXY.core.raster import (
    normalize_within_cams_cells,
    rasterize_country_ids,
    validate_weight_sums,
    warp_raster_to_ref,
    write_multiband_geotiff,
)

# Pixel chunk size for **post-main** raveled scans over full ref grids (CAMS cell
# id + per-group ``used_pop_fallback``) when building ``*_pop_proxy_fallback_by_cell.csv``.
# Processing in slices avoids holding or repeatedly indexing the entire flat array
# in one shot on large windows; it does **not** change numeric results, only how
# much memory is touched at a time. Also passed to ``max_valid_cam_cell_id`` so
# that helper can bound work similarly on huge rasters.
_CHUNK = 2_000_000


def _debug_log_alpha_vs_pollutants(
    alpha: np.ndarray,
    pollutants: list[str],
    iso3_list: list[str],
    fb_iso: str,
    *,
    sector_key: str,
    logger: logging.Logger,
) -> None:
    """Log whether CEIP alpha rows differ across pollutants for the focus country (catches CEIP/YAML bugs)."""
    iso_u = str(fb_iso).strip().upper()
    if iso_u not in iso3_list:
        return
    ri = iso3_list.index(iso_u)
    pol_lower = [str(p).lower() for p in pollutants]
    idx = {name: j for j, name in enumerate(pol_lower)}

    # Pairs users compare side-by-side in maps
    for a, b in (
        ("co", "nmvoc"),
        ("sox", "pm10"),
        ("pm10", "pm2_5"),
        ("nox", "nh3"),
        ("ch4", "co"),
    ):
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        row1 = np.asarray(alpha[ri, :, ia], dtype=np.float64).ravel()
        row2 = np.asarray(alpha[ri, :, ib], dtype=np.float64).ravel()
        dmax = float(np.max(np.abs(row1 - row2)))
        logger.info(
            "[%s] alpha CEIP row %s: group-share max_abs_diff(%s vs %s)=%.6g",
            sector_key,
            iso_u,
            pollutants[ia],
            pollutants[ib],
            dmax,
        )

    # Worst pair among all pollutants (single scalar — catches accidental duplicate slices)
    n_pol = len(pollutants)
    worst = 0.0
    worst_pair = ("", "")
    for j1 in range(n_pol):
        for j2 in range(j1 + 1, n_pol):
            row1 = np.asarray(alpha[ri, :, j1], dtype=np.float64).ravel()
            row2 = np.asarray(alpha[ri, :, j2], dtype=np.float64).ravel()
            dmax = float(np.max(np.abs(row1 - row2)))
            if dmax > worst:
                worst = dmax
                worst_pair = (pollutants[j1], pollutants[j2])
    logger.info(
        "[%s] alpha CEIP row %s: worst_pair_any_pollutants=%s max_abs_diff=%.6g",
        sector_key,
        iso_u,
        worst_pair,
        worst,
    )


def _debug_log_weight_band_separation(
    bands_out: dict[str, np.ndarray],
    pollutants: list[str],
    *,
    sector_key: str,
    logger: logging.Logger,
) -> None:
    """Log pairwise differences between final weight rasters (detect ×α bugs vs proportional bands)."""
    names = [p for p in pollutants if p in bands_out]
    if len(names) < 2:
        return
    ref_name = names[0]
    ref = np.asarray(bands_out[ref_name], dtype=np.float64)
    finite_ref = np.isfinite(ref) & (ref > 0)
    scale = float(np.nanpercentile(np.abs(ref[finite_ref]), 98)) if np.any(finite_ref) else 1.0
    scale = max(scale, 1e-30)
    for other in names[1:]:
        b = np.asarray(bands_out[other], dtype=np.float64)
        if b.shape != ref.shape:
            logger.warning(
                "[%s] band shape mismatch %s %s vs ref %s",
                sector_key,
                other,
                b.shape,
                ref.shape,
            )
            continue
        diff = np.nanmax(np.abs(ref - b))
        rel = diff / scale
        # Nearly proportional maps → identical Folium percentile colours
        cor = np.nan
        m = np.isfinite(ref) & np.isfinite(b) & (ref != 0.0) & (b != 0.0)
        if np.any(m):
            r0 = ref[m].ravel()
            b0 = b[m].ravel()
            if np.std(r0) > 1e-30 and np.std(b0) > 1e-30:
                cor = float(np.corrcoef(r0, b0)[0, 1])
        logger.info(
            "[%s] weight band max_abs_diff(%s vs %s)=%.6g (vs P98|ref|≈%.6g → rel=%.4g); corr≈%.5g",
            sector_key,
            ref_name,
            other,
            diff,
            scale,
            rel,
            cor,
        )
        if rel < 1e-5 and cor > 0.9999:
            logger.warning(
                "[%s] bands %s and %s are almost identical (relative diff %.3g, corr %.5g). "
                "Percentile map colours will match; check CEIP alphas or whether only one group dominates spatially.",
                sector_key,
                ref_name,
                other,
                rel,
                cor,
            )

    # Explicit pairs users compare in Folium / bbox plots
    for a, b in (
        ("co", "nmvoc"),
        ("sox", "pm10"),
        ("pm10", "pm2_5"),
        ("nox", "nh3"),
    ):
        if a not in bands_out or b not in bands_out:
            continue
        ra = np.asarray(bands_out[a], dtype=np.float64)
        rb = np.asarray(bands_out[b], dtype=np.float64)
        diff = np.nanmax(np.abs(ra - rb))
        cor = np.nan
        m = np.isfinite(ra) & np.isfinite(rb) & (ra != 0.0) & (rb != 0.0)
        if np.any(m):
            x0, y0 = ra[m].ravel(), rb[m].ravel()
            if np.std(x0) > 1e-30 and np.std(y0) > 1e-30:
                cor = float(np.corrcoef(x0, y0)[0, 1])
        logger.info(
            "[%s] weight pair (%s vs %s): max_abs_diff=%.6g corr≈%.5g",
            sector_key,
            a,
            b,
            diff,
            cor,
        )


# Used when a sector YAML does not set ``pollutants:`` (CEIP + multiband output order).
DEFAULT_CEIP_GROUP_POLLUTANTS: tuple[str, ...] = (
    "ch4",
    "co",
    "nh3",
    "nmvoc",
    "nox",
    "pm10",
    "pm2_5",
    "sox",
)


def _iso3_upper(value: Any, *, default: str = "GRC") -> str:
    """Normalize CAMS / CEIP country token to upper-case ISO-3 (empty -> ``default``)."""
    s = str(value or default).strip().upper()
    return s if s else default


def _first_nonempty_dict(*candidates: Any) -> dict[str, Any]:
    """Return a shallow copy of the first non-empty dict among ``candidates`` (same ``or`` chain semantics)."""
    for c in candidates:
        if isinstance(c, dict) and len(c) > 0:
            return dict(c)
    return {}


def merge_ceip_group_sector_cfg(
    *,
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    sector_paths_key: str,
    default_groups_yaml: str,
    osm_key: str,
    output_path: Path,
    default_pollutants: tuple[str, ...] = DEFAULT_CEIP_GROUP_POLLUTANTS,
) -> dict[str, Any]:
    """
    Merge path config + sector YAML into the structure consumed by
    :func:`run_gnfr_group_pipeline` (B_Industry, D_Fugitive).

    * ``default_pollutants`` applies only if ``sector_cfg`` has no ``pollutants`` key
      (``industry.yaml`` / ``fugitive.yaml`` normally define it explicitly;
      :data:`DEFAULT_CEIP_GROUP_POLLUTANTS` is the cross-sector fallback order).
    * ``group_order`` in the merged dict comes from ``ceip_group_order`` or
      ``group_order`` in sector YAML (list of keys matching ``groups:`` in the CEIP
      profile); if unset, :data:`PROXY.core.ceip.DEFAULT_GNFR_GROUP_ORDER` is used.
    """
    sp = sector_cfg.get(sector_paths_key) or {}
    ceip_ov = sector_cfg.get("ceip") if isinstance(sector_cfg.get("ceip"), dict) else {}
    pcommon = path_cfg.get("proxy_common") or {}
    alpha_w = (
        ceip_ov.get("workbook")
        or ceip_ov.get("ceip_workbook")
        or sp.get("ceip_workbook")
        or pcommon.get("alpha_workbook")
    )
    if not alpha_w:
        raise ValueError(
            f"{sector_paths_key}: set {sector_paths_key}.ceip_workbook or paths.proxy_common.alpha_workbook."
        )
    ceip_workbook = resolve_path(root, alpha_w)
    _groups_rel = str(
        ceip_ov.get("groups_yaml")
        or ceip_ov.get("ceip_groups_yaml")
        or sp.get("ceip_groups_yaml", default_groups_yaml)
    )
    ceip_groups = resolve_path(root, remap_legacy_ceip_relpath(_groups_rel))
    paths_dict: dict[str, Any] = {
        "cams_nc": resolve_path(root, path_cfg["emissions"]["cams_2019_nc"]),
        "corine": resolve_path(root, path_cfg["proxy_common"]["corine_tif"]),
        "population_tif": resolve_path(root, path_cfg["proxy_common"]["population_tif"]),
        "nuts_gpkg": resolve_path(root, path_cfg["proxy_common"]["nuts_gpkg"]),
        "ceip_workbook": ceip_workbook,
        "ceip_groups_yaml": ceip_groups,
        "ceip_years": ceip_ov.get("years")
        or ceip_ov.get("ceip_years")
        or sp.get("ceip_years", sector_cfg.get("ceip_years")),
        "osm_group_gpkg": resolve_path(root, path_cfg["osm"][osm_key]),
    }
    for opt_key in ("viirs_nightfire_csv", "gcmt_xlsx", "goget_xlsx"):
        rel = sp.get(opt_key)
        if rel:
            paths_dict[opt_key] = resolve_path(root, rel)
    pollutants = sector_cfg.get("pollutants") or list(default_pollutants)
    out_extra = sector_cfg.get("output") or {}
    raw_go = (
        sector_cfg.get("ceip_group_order")
        or sector_cfg.get("group_order")
        or ceip_ov.get("group_order")
    )
    if raw_go is not None:
        group_order = tuple(str(x).strip() for x in raw_go if str(x).strip())
        if not group_order:
            group_order = DEFAULT_GNFR_GROUP_ORDER
    else:
        group_order = DEFAULT_GNFR_GROUP_ORDER
    return {
        "_project_root": root,
        "defaults": {
            "fallback_country_iso3": _iso3_upper(sector_cfg.get("cams_country_iso3")),
        },
        "paths": paths_dict,
        "cntr_code_to_iso3": _first_nonempty_dict(
            ceip_ov.get("cntr_code_to_iso3"),
            sp.get("cntr_code_to_iso3"),
            sector_cfg.get("cntr_code_to_iso3"),
        ),
        "output": {
            "dir": str(output_path.parent.resolve()),
            "weights_tif": output_path.name,
            "write_debug_rasters": bool(out_extra.get("write_debug_rasters", False)),
        },
        "pollutants": [str(p) for p in pollutants],
        "ceip_pollutant_aliases": dict(
            ceip_ov.get("pollutant_aliases")
            or sector_cfg.get("ceip_pollutant_aliases")
            or {}
        ),
        "proxy": dict(sector_cfg.get("proxy") or {}),
        "logging": dict(sector_cfg.get("logging") or {"level": "WARNING"}),
        "group_order": group_order,
    }


def _file_mb(p: Path) -> float:
    try:
        return float(p.stat().st_size) / (1024.0 * 1024.0)
    except OSError:
        return -1.0


def _ensure_pipeline_log_handler(logger: logging.Logger) -> None:
    if logger.handlers:
        return
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = True


def _ensure_corine_path(root: Path, cfg: dict[str, Any], ref: dict[str, Any]) -> Path:
    p = ref.get("corine_path")
    if p is not None and Path(p).is_file():
        return Path(p)
    rel = (cfg.get("paths") or {}).get("corine")
    if rel:
        q = resolve_path(root, rel)
        if q.is_file():
            return q
    raise FileNotFoundError("CORINE path not set on ref and could not be resolved from config.")


def _write_debug_rasters(
    out_dir: Path,
    ref: dict[str, Any],
    group_pg: dict[str, dict[str, np.ndarray]],
    w_by_g: dict[str, np.ndarray],
    *,
    group_order: tuple[str, ...],
) -> None:
    kw = {
        "driver": "GTiff",
        "height": int(ref["height"]),
        "width": int(ref["width"]),
        "count": 1,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": ref["transform"],
        "compress": "deflate",
        "tiled": True,
    }
    for gid in group_order:
        d = group_pg.get(gid) or {}
        for name, arr in (
            ("proxy_osm", d.get("osm_raw")),
            ("proxy_clc", d.get("clc_raw")),
            ("proxy_p_g", d.get("p_g")),
        ):
            if arr is None:
                continue
            safe = str(gid).lower().replace(" ", "_").replace("/", "_")
            fp = out_dir / f"{name}_{safe}.tif"
            with rasterio.open(fp, "w", **kw) as dst:
                dst.write(np.asarray(arr, dtype=np.float32), 1)
    for gid in group_order:
        w = w_by_g.get(gid)
        if w is None:
            continue
        safe = str(gid).lower().replace(" ", "_").replace("/", "_")
        fp = out_dir / f"proxy_w_{safe}.tif"
        with rasterio.open(fp, "w", **kw) as dst:
            dst.write(np.asarray(w, dtype=np.float32), 1)


def _load_default_osm(path: Path) -> Any:
    import geopandas as gpd

    return gpd.read_file(path)


def _read_industry_gpkg_all_layers(path: Path) -> Any:
    import fiona
    import geopandas as gpd

    try:
        raw = fiona.listlayers(str(path))
    except fiona.errors.DriverError:
        return gpd.read_file(path)
    names: list[str] = []
    for e in raw:
        if isinstance(e, (list, tuple)) and e:
            names.append(str(e[0]))
        else:
            names.append(str(e))
    if len(names) <= 1:
        return gpd.read_file(path, layer=names[0] if names else None)
    parts: list[Any] = []
    for n in names:
        try:
            g = gpd.read_file(path, layer=n)
        except fiona.errors.DriverError:
            continue
        if g is not None and not g.empty:
            parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    crs0 = parts[0].crs
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True, sort=False), crs=crs0)


def run_gnfr_group_pipeline(
    *,
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    sector_key: str,
    output_prefix: str,
    country_iso3_fallback: str,
    show_progress: bool,
    logger: logging.Logger,
    diag_tag: str,
    osm_loader: Callable[[Path], Any] | None = None,
) -> Path:
    """
    Shared GNFR **group** pipeline (B industry, D fugitive): CEIP α tensor × per-group
    spatial proxy, one multiband GeoTIFF per pollutant.

    Stages: CAMS cell id → warp CORINE + pop → load OSM (optional multi-layer) →
    :func:`build_all_group_pg` (OSM/CORINE/sector score + population blend) →
    per-group normalize within cells → for each pollutant band, sum over groups
    (α[country, g, p] · w_g) and normalize that band in cells. Writes weights TIF
    and CSV sidecars under the output directory from ``cfg``.

    ``cfg["group_order"]`` lists CEIP group keys (order = tensor axis / alpha columns).
    """
    from PROXY.core.ceip.reported_group_alpha import load_ceip_and_alpha
    # Prepare paths and settings for the shared GNFR group pipeline.
    # - Extract necessary file paths and output directory from the configuration.
    # - Determine CEIP group order and validate its uniqueness.
    # - Compute output directory and file name, create the output directory if needed.
    # - Resolve the fallback country ISO3 code for CEIP/NUTS grid mapping.
    paths = cfg["paths"]
    group_order = tuple(str(x).strip() for x in (cfg.get("group_order") or ()))
    if not group_order:
        group_order = DEFAULT_GNFR_GROUP_ORDER
    if len(set(group_order)) != len(group_order):
        raise ValueError(f"group_order must have unique entries, got {group_order!r}")
    out_block = cfg.get("output") or {}
    out_dir = resolve_path(root, out_block.get("dir", "SourceProxies/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = str(out_block.get("weights_tif", f"{output_prefix}_areasource.tif"))
    out_tif = out_dir / out_name
    fb_iso = (
        country_iso3_fallback
        or (cfg.get("defaults") or {}).get("fallback_country_iso3")
        or "GRC"
    ).strip().upper()
    # --- Main input file resolution and configuration section ---
    # Resolves all input file paths needed for the GNFR group pipeline:
    #   - NetCDF of CAMS reference grid
    #   - CORINE land cover raster
    #   - Population raster
    #   - OSM industry/group GeoPackage
    #   - NUTS administrative boundaries vector
    #   - CEIP groups YAML (group definitions)
    #   - CEIP reported values workbook
    nc_path = resolve_path(root, paths["cams_nc"])
    corine_path = _ensure_corine_path(root, cfg, ref)
    pop_path = resolve_path(root, paths["population_tif"])
    osm_path = resolve_path(root, paths["osm_group_gpkg"])
    nuts_path = resolve_path(root, paths["nuts_gpkg"])
    groups_yaml = resolve_path(
        root, remap_legacy_ceip_relpath(str(paths["ceip_groups_yaml"]))
    )
    ceip_workbook = resolve_path(root, paths["ceip_workbook"])

    # Get proxy config and diagnostics logger
    pcfg = cfg.get("proxy") or {}
    log_stats = bool(pcfg.get("log_input_stats", True))
    diag = RasterStatsLogger(logger, tag=diag_tag)

    # Log start of the sector workflow, showing resolved paths, output locations, etc
    logger.info(
        "[%s] sector=%s: pipeline start (output_prefix=%r, out_tif=%s, country_iso3_fallback=%s, "
        "show_progress=%s — stage logs always at INFO; detailed raster stats if proxy.log_input_stats)",
        diag_tag,
        sector_key,
        output_prefix,
        out_tif,
        fb_iso,
        show_progress,
    )

    # Conditional: If log_stats is enabled, print input stats (shape, size, file existence)
    if log_stats:
        _ensure_pipeline_log_handler(logger)
        logger.info(
            "[%s] ref: height=%s width=%s crs=%s",
            diag_tag,
            ref.get("height"),
            ref.get("width"),
            ref.get("crs"),
        )
        for label, pth in (
            ("CAMS NC", nc_path),
            ("CORINE", corine_path),
            ("population_tif", pop_path),
            ("OSM GPKG", osm_path),
            ("NUTS", nuts_path),
            ("CEIP workbook", ceip_workbook),
            ("groups YAML", groups_yaml),
        ):
            logger.info(
                "[%s] %s: exists=%s size_MiB=%.3f path=%s",
                diag_tag,
                label,
                pth.is_file(),
                _file_mb(pth),
                pth,
            )

    # [1/8] Build CAMS cell id raster referenced to the uniform output grid
    logger.info(
        "%s pipeline [1/8]: build CAMS cell id raster (NetCDF + ref grid)…", output_prefix
    )
    cam_cell_id = build_cam_cell_id(nc_path, ref)

    # [2/8] Warp CORINE and population to the reference grid, build p_pop (proxy for population in CAMS cells)
    logger.info(
        "%s pipeline [2/8]: warp CORINE + population to reference, build p_pop…", output_prefix
    )
    clc = warp_raster_to_ref(
        corine_path,
        ref,
        band=1,
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=np.nan,
    )
    clc_nn = np.full(clc.shape, -9999, dtype=np.int32)
    m = np.isfinite(clc)
    clc_nn[m] = np.rint(clc[m]).astype(np.int32, copy=False)
    if log_stats:
        cam_ok = cam_cell_id >= 0
        diag.log_raster_stats("CLC (warped, float)", clc)
        diag.log_raster_stats("CLC_nn (int codes, -9999 nodata)", clc_nn.astype(np.float64), mask=cam_ok)

    pop = warp_raster_to_ref(
        pop_path,
        ref,
        band=1,
        resampling=Resampling.bilinear,
        src_nodata=None,
        dst_nodata=np.nan,
    )
    p_pop = build_p_pop(pop, ref)
    if log_stats:
        diag.log_population_chain(pop, ref, p_pop, cam_cell_id)

    # [3/8] Load OSM vector input (can be multi-layer for industry), check for valid CRS, decompostion of layers between different groups is done later
    logger.info(
        "%s pipeline [3/8]: load OSM GeoPackage (multi-layer for industry when applicable)…",
        output_prefix,
    )
    loader = osm_loader or _load_default_osm
    osm_gdf = loader(osm_path)
    if osm_gdf.crs is None:
        raise ValueError(f"OSM GeoPackage has no CRS: {osm_path}")

    # Load group definitions from YAML and verify all required groups are present
    with groups_yaml.open(encoding="utf-8") as f:
        groups_raw: dict[str, Any] = dict((yaml.safe_load(f) or {}).get("groups") or {})
    # Main GNFR group pipeline core: 
    #  - Verifies group definitions (YAML) contain all needed keys
    #  - Rasterizes NUTS polygons to country ID/ISO3 reference index, logs, picks fallback
    #  - Loads CEIP emission data and builds the group × pollutant "alpha" tensor
    #  - Normalizes per-group spatial proxy scores (computed from OSM, CORINE, population)
    #  - Creates pollutant-weighted bands for output, normalizes within CAMS cells
    #  - Writes output raster and several CSV diagnostics (alpha, fallback, proxy)
    #  - Optionally, writes debug rasters if configured

    # --- Verify GNFR groups YAML has all required group_order entries
    groups_keys = {str(k) for k in groups_raw}
    missing = [g for g in group_order if g not in groups_keys]
    if missing:
        raise ValueError(
            f"groups YAML {groups_yaml} has no definitions for group_order entries: {missing!r}; "
            f"available keys: {sorted(groups_keys)}"
        )

    # --- Rasterize NUTS regions and prepare ISO3 index for country coding
    country_id, iso3_for_index = rasterize_country_ids(nuts_path, ref)
    iso3_list = [str(iso3_for_index[i]).strip().upper() for i in range(1, len(iso3_for_index))]
    if not iso3_list:
        raise ValueError("No countries in NUTS rasterization list.")
    logger.info(
        "%s pipeline [4/8]: NUTS/ISO3 — %d countries in alpha list (e.g. first few: %s)",
        output_prefix,
        len(iso3_list),
        ", ".join(iso3_list[:5]) + ("…" if len(iso3_list) > 5 else ""),
    )
    # Fallback index for country mapping (for pixels with unmapped NUTS)
    fb_ri = iso3_list.index(fb_iso) if fb_iso in iso3_list else 0
    ri = np.where(country_id.astype(np.int64) > 0, country_id.astype(np.int64) - 1, fb_ri).astype(np.int64)
    ri = np.clip(ri, 0, len(iso3_list) - 1)
    if log_stats:
        diag.log_cam_and_country(cam_cell_id, country_id)

    # --- Load CEIP emissions workbook/group config, build (country, group, pollutant) alpha tensor
    logger.info(
        "%s pipeline [5/8]: load CEIP workbook, build group × pollutant alpha tensor…", output_prefix
    )
    alpha, _fb_code, wide_alpha = load_ceip_and_alpha(
        cfg, iso3_list, sector_key=sector_key, focus_country_iso3=fb_iso
    )
    alpha = np.asarray(alpha, dtype=np.float32)
    alpha_cols = tuple(f"alpha_{g}" for g in group_order)
    # Log the alpha tensor to a CSV file for debugging and analysis
    log_wide_group_alpha_table(
        logger,
        sector=sector_key,
        wide=wide_alpha,
        focus_iso3=fb_iso,
        group_cols=alpha_cols,
    )

    # --- Proxy computation for each group: combine OSM, CORINE, and population as spatial weights
    pollutants = [str(p) for p in cfg["pollutants"]]
    if os.environ.get("PROXY_DEBUG_ALPHA_WEIGHTS", "").strip().lower() in ("1", "true", "yes"):
        _debug_log_alpha_vs_pollutants(
            alpha,
            pollutants,
            iso3_list,
            fb_iso,
            sector_key=str(sector_key),
            logger=logger,
        )
    if str(sector_key) == "D_Fugitive":
        logger.info(
            "%s pipeline [6/8]: build per-group fugitive proxy (OSM+CLC+pop + VIIRS/GCMT/GOGET when configured) "
            "for %s…",
            output_prefix,
            ", ".join(group_order),
        )
    else:
        logger.info(
            "%s pipeline [6/8]: build per-group OSM+CORINE+population proxy scores (%s)…",
            output_prefix,
            ", ".join(group_order),
        )
    if str(sector_key) == "D_Fugitive":
        group_pg = build_fugitive_group_pg(
            clc_nn,
            osm_gdf,
            {"groups": groups_raw},
            ref,
            pcfg,
            p_pop,
            group_order=group_order,
            root=root,
            cfg=cfg,
            cam_cell_id=cam_cell_id,
        )
    else:
        group_pg = build_all_group_pg(
            clc_nn,
            osm_gdf,
            {"groups": groups_raw},
            ref,
            pcfg,
            p_pop,
            group_order=group_order,
            cam_cell_id=cam_cell_id,
        )

    # --- Normalize each group’s proxy score within CAMS cells (to sum to 1 inside each air cell)
    logger.info(
        "%s pipeline [7/8]: normalize each group’s proxy within CAMS cells (%d groups: %s)…",
        output_prefix,
        len(group_order),
        ", ".join(group_order),
    )
    w_by_g: dict[str, np.ndarray] = {}
    fb_summary: list[tuple[str, int]] = []
    for gid in group_order:
        p_g = group_pg[gid]["p_g"]
        w_n, _ = normalize_within_cams_cells(
            p_g,
            cam_cell_id,
            None,
            return_fallback_mask=False,
            context=f"{sector_key} group={gid} stage=per_group",
            uniform_fallback_summary=fb_summary,
        )
        w_by_g[gid] = w_n
        group_pg[gid]["p_g"] = w_n

    # --- For each pollutant, sum alpha × group_proxy for all groups, then normalize result
    h, w = int(ref["height"]), int(ref["width"])
    acc = np.zeros((h, w), dtype=np.float32)
    tmp = np.empty((h, w), dtype=np.float32)
    bands_out: dict[str, np.ndarray] = {}
    logger.info(
        "%s pipeline [8/8]: for each of %d pollutant bands, sum alpha×group weight and normalize in cells…",
        output_prefix,
        len(pollutants),
    )
    for j, name in enumerate(pollutants):
        acc.fill(0.0)
        for gi, gid in enumerate(group_order):
            np.multiply(alpha[ri, gi, j], w_by_g[gid], out=tmp)
            np.add(acc, tmp, out=acc)
        band, _ = normalize_within_cams_cells(
            acc,
            cam_cell_id,
            None,
            return_fallback_mask=False,
            context=f"{sector_key} pollutant={name} stage=band",
            uniform_fallback_summary=fb_summary,
        )
        errs = validate_weight_sums(band, cam_cell_id, None, tol=1e-3)
        if errs:
            logger.warning(
                "Weight sum check band %s: %d cells off (showing up to 5): %s",
                name,
                len(errs),
                errs[:5],
            )
        # Must copy: ``normalize_within_cams_cells`` mutates its buffer in-place and often
        # returns the same float32 C-contiguous array as ``acc``. ``np.ascontiguousarray``
        # does *not* copy in that case, so every ``bands_out`` entry would alias ``acc``
        # and all bands would match the last pollutant only.
        bands_out[name] = np.asarray(band, dtype=np.float32).copy()

    if os.environ.get("PROXY_DEBUG_ALPHA_WEIGHTS", "").strip().lower() in ("1", "true", "yes"):
        _debug_log_weight_band_separation(
            bands_out,
            pollutants,
            sector_key=str(sector_key),
            logger=logger,
        )

    # --- Write geotiff output, diagnostic CSVs, and optionally debug rasters
    write_multiband_geotiff(
        out_tif,
        bands_out,
        ref=ref,
        dtype="float32",
        nodata=None,
        tiled=True,
        sector=sector_key,
        pollutants=pollutants,
    )
    logger.info(
        "[%s] sector=%s: wrote multiband weights TIF -> %s", diag_tag, sector_key, out_tif
    )

    wide_alpha.to_csv(out_dir / f"{output_prefix}_alpha_country_pollutant.csv", index=False)
    fb_counts = wide_alpha.groupby("fallback_code").size().reset_index(name="n_rows")
    fb_counts.to_csv(out_dir / f"{output_prefix}_fallback_counts.csv", index=False)

    # --- Write CSV with fallback statistics: for each group, count of pixels using OSM/CLC fallback per CAMS cell
    pop_fb_rows: list[dict[str, Any]] = []
    max_cam = max_valid_cam_cell_id(cam_cell_id, _CHUNK)
    if max_cam >= 0:
        flat_c = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
        n_pix = int(flat_c.size)
        ce = max(10_000, _CHUNK)
        for gid in group_order:
            ufb = group_pg[gid]["used_pop_fallback"].ravel()
            counts = np.zeros(max_cam + 1, dtype=np.int64)
            for s in range(0, n_pix, ce):
                e = min(n_pix, s + ce)
                cc = flat_c[s:e]
                u = ufb[s:e] > 0
                m2 = (cc >= 0) & u
                if np.any(m2):
                    counts += np.bincount(cc[m2], minlength=max_cam + 1)
            nz = np.flatnonzero(counts)
            for c in nz:
                ni = int(counts[int(c)])
                if ni > 0:
                    pop_fb_rows.append(
                        {
                            "cam_cell_id": int(c),
                            "group": gid,
                            "n_pixels_osm_clc_fallback": ni,
                        }
                    )
    pd.DataFrame(pop_fb_rows).to_csv(
        out_dir / f"{output_prefix}_pop_proxy_fallback_by_cell.csv", index=False
    )

    if bool(out_block.get("write_debug_rasters", False)):
        _write_debug_rasters(out_dir, ref, group_pg, w_by_g, group_order=group_order)
    logger.info(
        "[%s] sector=%s: pipeline done (CSVs under %s)", diag_tag, sector_key, out_dir
    )
    return out_tif


def load_industry_osm_all_layers(path: Path) -> Any:
    return _read_industry_gpkg_all_layers(path)
