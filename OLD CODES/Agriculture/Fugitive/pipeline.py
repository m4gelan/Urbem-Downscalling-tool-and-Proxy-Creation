"""End-to-end fugitive area-source weights: CEIP alphas, spatial proxies, CAMS-cell normalize."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling

from Waste.j_waste_weights.cams_grid import build_cam_cell_id
from Waste.j_waste_weights.country_raster import rasterize_country_ids
from Waste.j_waste_weights.io_utils import warp_raster_to_ref
from Waste.j_waste_weights.normalization import normalize_within_cams_cells, validate_weight_sums

from .ceip_fugitive import load_ceip_and_alpha
from .config_utils import resolve
from .proxy_fugitive import build_all_group_pg, build_p_pop

logger = logging.getLogger(__name__)

_GROUP_ORDER = ("G1", "G2", "G3", "G4")

_CHUNK = 2_000_000

_pipeline_debug_handler_attached = False


def _max_valid_cam_cell_id(cam_cell_id: np.ndarray, chunk_elems: int = _CHUNK) -> int:
    """Largest ``cam_cell_id`` on the fine grid where id >= 0 (chunked, no full masked copy)."""
    cid = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
    n = int(cid.size)
    ce = max(10_000, int(chunk_elems))
    mx = -1
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        m = cc >= 0
        if np.any(m):
            mx = max(mx, int(cc[m].max()))
    return mx


def _ensure_fugitive_pipeline_log_handler() -> None:
    """Attach a stderr handler so ``[fugitive-debug]`` lines show even when root logging is unset."""
    global _pipeline_debug_handler_attached
    if _pipeline_debug_handler_attached:
        return
    if logger.handlers:
        _pipeline_debug_handler_attached = True
        return
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    _pipeline_debug_handler_attached = True


def _file_mb(p: Path) -> float:
    try:
        return float(p.stat().st_size) / (1024.0 * 1024.0)
    except OSError:
        return -1.0


def _log_raster_stats(name: str, arr: np.ndarray, *, mask: np.ndarray | None = None) -> None:
    """Log dtype, shape, finite fraction, and min/max/mean (optionally only where ``mask`` is True)."""
    a = np.asarray(arr)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(a.shape)
        sel = m & np.isfinite(a)
    else:
        sel = np.isfinite(a)
    n = int(a.size)
    n_ok = int(np.count_nonzero(sel))
    if n_ok == 0:
        logger.warning("[fugitive-debug] %s: shape=%s dtype=%s — no finite values in selection", name, a.shape, a.dtype)
        return
    v = a[sel]
    logger.info(
        "[fugitive-debug] %s: shape=%s dtype=%s selection=%d/%d (%.2f%%) min=%.6g max=%.6g mean=%.6g",
        name,
        a.shape,
        a.dtype,
        n_ok,
        n,
        100.0 * n_ok / max(n, 1),
        float(np.min(v)),
        float(np.max(v)),
        float(np.mean(v)),
    )


def _log_population_chain(pop: np.ndarray, ref: dict[str, Any], p_pop: np.ndarray, cam_cell_id: np.ndarray) -> None:
    area = float(abs(float(ref["transform"][0]) * float(ref["transform"][4])))
    dens = np.where(np.isfinite(pop) & (pop >= 0), pop.astype(np.float64) / max(area, 1e-6), np.nan)
    cam_ok = cam_cell_id >= 0
    _log_raster_stats("population (warped raw)", pop, mask=cam_ok)
    _log_raster_stats("population_density (pop/area_m2)", dens, mask=cam_ok & np.isfinite(dens))
    _log_raster_stats("P_pop = z(pop_density)", p_pop, mask=cam_ok)
    nz = int(np.count_nonzero((p_pop > 0) & cam_ok))
    tot = int(np.count_nonzero(cam_ok))
    logger.info(
        "[fugitive-debug] P_pop positive pixels (cam_cell_id>=0): %d / %d (%.2f%%)",
        nz,
        tot,
        100.0 * nz / max(tot, 1),
    )


def _log_cam_and_country(cam_cell_id: np.ndarray, country_id: np.ndarray) -> None:
    cid = np.asarray(cam_cell_id).ravel()
    n = cid.size
    n_in = int(np.count_nonzero(cid >= 0))
    n_out = int(np.count_nonzero(cid < 0))
    logger.info(
        "[fugitive-debug] cam_cell_id: in_domain=%d (%.2f%%) outside=%d (%.2f%%)",
        n_in,
        100.0 * n_in / max(n, 1),
        n_out,
        100.0 * n_out / max(n, 1),
    )
    if n_in:
        logger.info(
            "[fugitive-debug] cam_cell_id range (valid): [%d, %d] unique≈%d",
            int(cid[cid >= 0].min()),
            int(cid[cid >= 0].max()),
            int(np.unique(cid[cid >= 0]).size),
        )
    ctry = np.asarray(country_id).ravel()
    n_ct = int(np.count_nonzero(ctry > 0))
    logger.info(
        "[fugitive-debug] country_id (NUTS raster): pixels with id>0: %d / %d (%.2f%%)",
        n_ct,
        n,
        100.0 * n_ct / max(n, 1),
    )


def _log_osm_gdf(tag: str, gdf: Any, path: Path) -> None:
    import geopandas as gpd

    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.warning("[fugitive-debug] %s: not a GeoDataFrame", tag)
        return
    try:
        b = tuple(gdf.total_bounds)
    except Exception:
        b = (float("nan"),) * 4
    ne = int((gdf.geometry.notna() & ~gdf.geometry.is_empty).sum()) if len(gdf) else 0
    logger.info(
        "[fugitive-debug] %s: path=%s rows=%d non_empty_geom≈%d crs=%s bounds=%s cols=%s",
        tag,
        path,
        len(gdf),
        ne,
        gdf.crs,
        b,
        list(gdf.columns)[:20],
    )


def _ensure_corine_path(root: Path, fugitive_cfg: dict[str, Any], ref: dict[str, Any]) -> Path:
    p = ref.get("corine_path")
    if p is not None and Path(p).is_file():
        return Path(p)
    try:
        from SourceProxies.grid import first_existing_corine
    except ImportError:
        first_existing_corine = None  # type: ignore[assignment]
    rel = (fugitive_cfg.get("paths") or {}).get("corine")
    if first_existing_corine is not None and rel:
        return first_existing_corine(root, rel)
    raise FileNotFoundError("CORINE path not set on ref and could not resolve from fugitive config.")


def _write_debug_rasters(
    out_dir: Path,
    ref: dict[str, Any],
    group_pg: dict[str, dict[str, np.ndarray]],
    w_by_g: dict[str, np.ndarray],
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
    for gid in _GROUP_ORDER:
        d = group_pg.get(gid) or {}
        for name, arr in (
            ("proxy_osm", d.get("osm_raw")),
            ("proxy_clc", d.get("clc_raw")),
            ("proxy_p_g", d.get("p_g")),
        ):
            if arr is None:
                continue
            fp = out_dir / f"{name}_{gid.lower()}.tif"
            with rasterio.open(fp, "w", **kw) as dst:
                dst.write(np.asarray(arr, dtype=np.float32), 1)
    for gid in _GROUP_ORDER:
        w = w_by_g.get(gid)
        if w is None:
            continue
        fp = out_dir / f"proxy_w_{gid.lower()}.tif"
        with rasterio.open(fp, "w", **kw) as dst:
            dst.write(np.asarray(w, dtype=np.float32), 1)


def run_fugitive_pipeline(
    root: Path,
    fugitive_cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    country_iso3_fallback: str | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Write multi-band fugitive weights GeoTIFF and CSV diagnostics under ``output.dir``.

    ``ref`` must match the target fine grid (height, width, transform, crs).
    """
    paths = fugitive_cfg["paths"]
    out_block = fugitive_cfg.get("output") or {}
    out_dir = resolve(root, Path(out_block.get("dir", "SourceProxies/outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = str(out_block.get("weights_tif", "Fugitive_areasource.tif"))
    out_tif = out_dir / out_name

    defaults = fugitive_cfg.get("defaults") or {}
    fb_iso = (country_iso3_fallback or defaults.get("fallback_country_iso3") or "GRC").strip().upper()

    nc_path = resolve(root, Path(paths["cams_nc"]))
    corine_path = _ensure_corine_path(root, fugitive_cfg, ref)
    pop_path = resolve(root, Path(paths["population_tif"]))
    osm_path = resolve(root, Path(paths["osm_fugitive_gpkg"]))
    nuts_path = resolve(root, Path(paths["nuts_gpkg"]))
    groups_yaml = resolve(root, Path(paths["ceip_groups_yaml"]))
    ceip_xlsx = resolve(root, Path(paths["ceip_xlsx"]))

    pcfg = fugitive_cfg.get("proxy") or {}
    log_stats = bool(pcfg.get("log_input_stats", True))
    if log_stats:
        _ensure_fugitive_pipeline_log_handler()
        logger.info(
            "[fugitive-debug] ref: height=%s width=%s crs=%s",
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
            ("CEIP xlsx", ceip_xlsx),
            ("groups YAML", groups_yaml),
        ):
            logger.info(
                "[fugitive-debug] %s: exists=%s size_MiB=%.3f path=%s",
                label,
                pth.is_file(),
                _file_mb(pth),
                pth,
            )

    if show_progress:
        logger.info("Fugitive pipeline: CAMS cell ids…")
    cam_cell_id = build_cam_cell_id(nc_path, ref)

    if show_progress:
        logger.info("Fugitive pipeline: warping CORINE + population…")
    clc = warp_raster_to_ref(
        corine_path,
        ref,
        band=1,
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=np.nan,
    )
    clc_nn = np.full(clc.shape, -9999, dtype=np.int32)
    _m = np.isfinite(clc)
    clc_nn[_m] = np.rint(clc[_m]).astype(np.int32, copy=False)
    if log_stats:
        cam_ok = cam_cell_id >= 0
        _log_raster_stats("CLC (warped, float)", clc)
        _log_raster_stats("CLC_nn (int codes, -9999 nodata)", clc_nn.astype(np.float64), mask=cam_ok & (clc_nn != -9999))
        vm = clc_nn[clc_nn != -9999]
        if vm.size and int(vm.max()) <= 99:
            logger.info(
                "[fugitive-debug] CORINE uses CLC 1–44 style codes (max=%d). "
                "YAML Level-2 classes (121, 131, …) are mapped to 44-class indices in proxy_fugitive.",
                int(vm.max()),
            )

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
        _log_population_chain(pop, ref, p_pop, cam_cell_id)

    if show_progress:
        logger.info("Fugitive pipeline: loading OSM GPKG…")
    import geopandas as gpd

    osm_gdf = gpd.read_file(osm_path)
    if osm_gdf.crs is None:
        raise ValueError(f"OSM GeoPackage has no CRS: {osm_path}")
    if log_stats:
        _log_osm_gdf("OSM fugitive (full layer)", osm_gdf, osm_path)

    with groups_yaml.open(encoding="utf-8") as f:
        group_specs_root = yaml.safe_load(f) or {}
    groups_raw: dict[str, Any] = dict(group_specs_root.get("groups") or {})

    if show_progress:
        logger.info("Fugitive pipeline: country raster…")
    country_id, iso3_for_index = rasterize_country_ids(nuts_path, ref)
    iso3_list = [str(iso3_for_index[i]).strip().upper() for i in range(1, len(iso3_for_index))]
    if not iso3_list:
        raise ValueError("No countries in NUTS rasterization list.")

    try:
        fb_ri = iso3_list.index(fb_iso)
    except ValueError:
        logger.warning("fallback_country_iso3 %s not in NUTS list; using index 0 (%s).", fb_iso, iso3_list[0])
        fb_ri = 0

    ri = np.where(country_id.astype(np.int64) > 0, country_id.astype(np.int64) - 1, fb_ri).astype(np.int64)
    ri = np.clip(ri, 0, len(iso3_list) - 1)
    if log_stats:
        _log_cam_and_country(cam_cell_id, country_id)

    if show_progress:
        logger.info("Fugitive pipeline: CEIP alphas…")
    alpha, _fb_code, wide_alpha = load_ceip_and_alpha(fugitive_cfg, iso3_list)
    # Keep alphas in float32 so ``alpha[ri, gi, j]`` does not allocate a full-grid float64 view (~500 MiB).
    alpha = np.asarray(alpha, dtype=np.float32)
    if log_stats:
        logger.info(
            "[fugitive-debug] CEIP alpha: shape=%s (countries x groups x pollutants) wide_alpha rows=%d",
            alpha.shape,
            len(wide_alpha),
        )

    pollutants = [str(p) for p in fugitive_cfg["pollutants"]]
    n_pol = len(pollutants)

    if show_progress:
        logger.info("Fugitive pipeline: group proxies P_g…")
    group_pg = build_all_group_pg(
        clc_nn,
        osm_gdf,
        {"groups": groups_raw},
        ref,
        pcfg,
        p_pop,
    )
    if log_stats:
        cam_ok = cam_cell_id >= 0
        for gid in _GROUP_ORDER:
            d = group_pg[gid]
            _log_raster_stats(f"osm_raw[{gid}]", d["osm_raw"], mask=cam_ok)
            _log_raster_stats(f"clc_raw[{gid}]", d["clc_raw"], mask=cam_ok)
            _log_raster_stats(f"P_g[{gid}]", d["p_g"], mask=cam_ok)
            nufb = int(np.count_nonzero(d["used_pop_fallback"].astype(bool) & cam_ok))
            logger.info(
                "[fugitive-debug] OSM+CLC→pop fallback pixels (cam ok) for %s: %d (%.2f%% of cam-ok pixels)",
                gid,
                nufb,
                100.0 * nufb / max(int(np.count_nonzero(cam_ok)), 1),
            )

    w_by_g: dict[str, np.ndarray] = {}
    for gid in _GROUP_ORDER:
        p_g = group_pg[gid]["p_g"]
        # Skip fallback mask (saves a full-grid bool). Use return value so a C-contig copy still lands in ``w_by_g``.
        w_n, _ = normalize_within_cams_cells(p_g, cam_cell_id, None, return_fallback_mask=False)
        w_by_g[gid] = w_n
        group_pg[gid]["p_g"] = w_n

    h, w = int(ref["height"]), int(ref["width"])
    acc = np.zeros((h, w), dtype=np.float32)
    tmp = np.empty((h, w), dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": n_pol,
        "dtype": "float32",
        "crs": ref["crs"],
        "transform": ref["transform"],
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        for j, name in enumerate(pollutants):
            acc.fill(0.0)
            for gi, gid in enumerate(_GROUP_ORDER):
                np.multiply(alpha[ri, gi, j], w_by_g[gid], out=tmp)
                np.add(acc, tmp, out=acc)
            band, _ = normalize_within_cams_cells(acc, cam_cell_id, None, return_fallback_mask=False)
            errs = validate_weight_sums(band, cam_cell_id, None, tol=1e-3)
            if errs:
                sample = errs[:5]
                logger.warning(
                    "Weight sum check band %s: %d cells off (showing up to 5): %s",
                    name,
                    len(errs),
                    sample,
                )
            dst.write(band, j + 1)
            dst.set_band_description(j + 1, name)

    wide_alpha.to_csv(out_dir / "fugitive_alpha_country_pollutant.csv", index=False)

    fb_counts = wide_alpha.groupby("fallback_code").size().reset_index(name="n_rows")
    fb_counts.to_csv(out_dir / "fugitive_fallback_counts.csv", index=False)

    pop_fb_rows: list[dict[str, Any]] = []
    max_cam = _max_valid_cam_cell_id(cam_cell_id, _CHUNK)
    if max_cam >= 0:
        flat_c = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
        n_pix = int(flat_c.size)
        ce = max(10_000, _CHUNK)
        for gid in _GROUP_ORDER:
            ufb = group_pg[gid]["used_pop_fallback"].ravel()
            counts = np.zeros(max_cam + 1, dtype=np.int64)
            for s in range(0, n_pix, ce):
                e = min(n_pix, s + ce)
                cc = flat_c[s:e]
                u = ufb[s:e] > 0
                m = (cc >= 0) & u
                if np.any(m):
                    counts += np.bincount(cc[m], minlength=max_cam + 1)
            nz = np.flatnonzero(counts)
            for c in nz:
                ni = int(counts[int(c)])
                if ni > 0:
                    pop_fb_rows.append(
                        {"cam_cell_id": int(c), "group": gid, "n_pixels_osm_clc_fallback": ni}
                    )
    pd.DataFrame(pop_fb_rows).to_csv(out_dir / "fugitive_pop_proxy_fallback_by_cell.csv", index=False)

    if bool(out_block.get("write_debug_rasters", False)):
        _write_debug_rasters(out_dir, ref, group_pg, w_by_g)

    logger.info("Wrote %s", out_tif)
    return out_tif
