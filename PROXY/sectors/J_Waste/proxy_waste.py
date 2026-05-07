"""
Build normalized proxy rasters P_solid, P_ww, P_res on the fine reference grid.

GIS assumptions are documented in module docstrings per function.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from shapely.ops import unary_union

from PROXY.core.dataloaders import resolve_path
from PROXY.core.osm_corine_proxy import adapt_corine_classes_for_grid
from PROXY.core.raster import warp_raster_to_ref

logger = logging.getLogger(__name__)

# waste_family values from data/OSM/build_osm_waste_layers.py::_waste_family — what each OSM group covers.
OSM_WASTE_FAMILY_COVER: dict[str, str] = {
    "landfill": "landuse=landfill (polygons)",
    "wastewater_plant": "man_made=wastewater_plant (polygons; excluded from the solid proxy)",
    "amenity_waste_disposal": "amenity=waste_disposal (polygons and buffered nodes)",
    "amenity_recycling": "amenity=recycling (polygons and buffered nodes)",
    "other": "other OSM waste-related polygons not classified above",
}

POLY_LAYER = "osm_waste_polygons"
POINT_LAYER = "osm_waste_points"


def _find_first_raster_in_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".jp2"):
        return path
    for pat in ("*.tif", "*.tiff", "*.jp2"):
        found = sorted(path.glob(pat))
        if found:
            return found[0]
    found = sorted(path.rglob("*.tif"))
    return found[0] if found else None


def _read_raster_path_or_dir(path: Path, ref: dict[str, Any], **kwargs: Any) -> np.ndarray:
    p = path
    if p.is_dir():
        inner = _find_first_raster_in_dir(p)
        if inner is None:
            raise FileNotFoundError(f"No raster found under directory: {p}")
        p = inner
        logger.info("Using raster from directory: %s", p)
    return warp_raster_to_ref(p, ref, **kwargs)


def _load_osm_waste_polygons_points(p: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read layers from osm_waste_layers.gpkg; fall back to a single layer with mixed geometry."""
    try:
        import fiona
    except ImportError:
        fiona = None  # type: ignore[assignment]
    poly = gpd.GeoDataFrame()
    pts = gpd.GeoDataFrame()
    if fiona is not None:
        try:
            layers = fiona.listlayers(str(p))
        except Exception:
            layers = []
        if POLY_LAYER in layers:
            poly = gpd.read_file(p, layer=POLY_LAYER)
        if POINT_LAYER in layers:
            pts = gpd.read_file(p, layer=POINT_LAYER)
    if poly.empty and pts.empty:
        g = gpd.read_file(p)
        if g.empty or "waste_family" not in g.columns:
            return poly, pts
        gt = g.geometry.geom_type
        pts = g[gt == "Point"].copy()
        poly = g[~gt.isin(["Point", "MultiPoint"])].copy()
    return poly, pts


def _osm_geometries_for_solid(
    g_poly: gpd.GeoDataFrame,
    g_pts: gpd.GeoDataFrame,
    crs: rasterio.crs.CRS,
    *,
    point_buffer_m: float,
) -> tuple[list, dict[str, list]]:
    """
    Project to ref CRS, buffer points, return (all geoms for combined mask) and
    {waste_family: [geoms]} for per-family masks.
    """
    geoms_all: list = []
    by_fam: dict[str, list] = defaultdict(list)
    if not g_poly.empty:
        gp = g_poly
        if gp.crs is None:
            logger.warning("OSM waste polygons have no CRS; assuming EPSG:3035 (Waste OSM GPKG convention).")
            gp = gp.set_crs("EPSG:3035")
        gp = gp.to_crs(crs)
        for _, row in gp.iterrows():
            geom = row.geometry
            fam = str(row.get("waste_family", "other") or "other")
            if geom is None or geom.is_empty:
                continue
            geoms_all.append(geom)
            by_fam[fam].append(geom)
    if not g_pts.empty:
        gt = g_pts
        if gt.crs is None:
            logger.warning("OSM waste points have no CRS; assuming EPSG:3035 (Waste OSM GPKG convention).")
            gt = gt.set_crs("EPSG:3035")
        gt = gt.to_crs(crs)
        for _, row in gt.iterrows():
            geom = row.geometry
            fam = str(row.get("waste_family", "other") or "other")
            if geom is None or geom.is_empty:
                continue
            b = geom.buffer(float(point_buffer_m))
            geoms_all.append(b)
            by_fam[fam].append(b)
    return geoms_all, dict(by_fam)

try:
    from scipy import ndimage as _ndimage

    _HAS_SCIPY = True
except Exception:
    _ndimage = None
    _HAS_SCIPY = False


def _quantile_minmax(
    arr: np.ndarray,
    q_low: float = 0.01,
    q_high: float = 0.99,
    *,
    max_samples: int = 2_000_000,
) -> np.ndarray:
    """
    Linear stretch using quantiles in float32 to limit memory.

    For very large finite sets, quantiles use a fixed-size random subsample (deterministic seed).
    """
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, dtype=np.float32)
    if finite.size > max_samples:
        rng = np.random.default_rng(42)
        finite = rng.choice(finite, size=max_samples, replace=False)
    lo = float(np.quantile(finite, q_low))
    hi = float(np.quantile(finite, q_high))
    if hi <= lo:
        hi = lo + 1e-6
    scale = np.float32(hi - lo)
    y = np.zeros(x.shape, dtype=np.float32)
    m = np.isfinite(x)
    y[m] = (x[m] - np.float32(lo)) / scale
    np.clip(y, 0.0, 1.0, out=y)
    return y


def _normalize(arr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    ncfg = (cfg.get("proxy") or {}).get("normalization") or {}
    method = str(ncfg.get("method", "quantile_minmax")).lower()
    if method == "quantile_minmax":
        return _quantile_minmax(
            arr,
            float(ncfg.get("q_low", 0.01)),
            float(ncfg.get("q_high", 0.99)),
        )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _rasterize_mask(
    geoms: list,
    ref: dict[str, Any],
    *,
    burn: float = 1.0,
    all_touched: bool = False,
) -> np.ndarray:
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    if not geoms:
        return np.zeros((h, w), dtype=np.float64)
    shapes = [(g, burn) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((h, w), dtype=np.float64)
    return features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0.0,
        dtype=np.float64,
        all_touched=all_touched,
    ).astype(np.float64)


def build_proxy_solid(cfg: dict[str, Any], ref: dict[str, Any]) -> np.ndarray:
    """
    Solid waste treatment proxy: CORINE 132 (dump sites) primary, 121 secondary;
    optional OSM GPKG (``osm_waste_polygons`` / ``osm_waste_points``) as additive refinement.

    Per OSM group semantics: see ``OSM_WASTE_FAMILY_COVER`` (aligned with
    ``data/OSM/build_osm_waste_layers.py``). Point amenities are buffered by ``osm_point_buffer_m``.

    Rasterization: polygon **coverage** is binary inside geometry
    (``all_touched=False``); not fractional sub-pixel area.
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("solid") or {}
    corine_path = Path(ref["corine_path"])
    if not corine_path.is_file():
        corine_path = resolve_path(root, Path(paths["corine"]))
    clc = warp_raster_to_ref(corine_path, ref, resampling=Resampling.nearest)
    # Phase 4.3 fix: previously this used literal equality against CORINE Level-3 codes
    # (``clc == 132`` / ``clc == 121``), which silently produced all-zero masks when the
    # warped raster held EEA44 indices (1..44) rather than raw L3 codes. Run both L3
    # codes through ``adapt_corine_classes_for_grid`` so the mask is correct in either
    # encoding.
    clc_f = np.asarray(clc, dtype=np.float64)
    clc_int = np.full(clc_f.shape, -9999, dtype=np.int32)
    mfin = np.isfinite(clc_f)
    clc_int[mfin] = np.rint(clc_f[mfin]).astype(np.int32, copy=False)
    codes_132, _ = adapt_corine_classes_for_grid(clc_int, [132])
    codes_121, _ = adapt_corine_classes_for_grid(clc_int, [121])
    m132 = np.isin(clc_int, np.asarray(codes_132, dtype=np.int32)).astype(np.float64)
    m121 = np.isin(clc_int, np.asarray(codes_121, dtype=np.int32)).astype(np.float64)
    w132 = float(pcfg.get("weight_clc_132", 1.0))
    w121 = float(pcfg.get("weight_clc_121", 0.35))
    base = w132 * _normalize(m132, cfg) + w121 * _normalize(m121, cfg)
    osm_path = paths.get("osm_waste_gpkg")
    buf_m = float(pcfg.get("osm_point_buffer_m", 50.0))
    if osm_path:
        p = Path(osm_path)
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            g_poly, g_pts = _load_osm_waste_polygons_points(p)
            if "waste_family" in g_poly.columns:
                g_poly = g_poly[g_poly["waste_family"].astype(str) != "wastewater_plant"].copy()
            if "waste_family" in g_pts.columns:
                g_pts = g_pts[g_pts["waste_family"].astype(str) != "wastewater_plant"].copy()
            crs = rasterio.crs.CRS.from_string(ref["crs"])
            geoms_all, by_fam = _osm_geometries_for_solid(
                g_poly, g_pts, crs, point_buffer_m=buf_m
            )
            if geoms_all:
                for fam, geoms in sorted(by_fam.items()):
                    cover = OSM_WASTE_FAMILY_COVER.get(fam, fam)
                    logger.info(
                        "OSM solid proxy: waste_family=%s (%s) — %d geometries",
                        fam,
                        cover,
                        len(geoms),
                    )
                r_osm = _rasterize_mask(geoms_all, ref, burn=1.0)
                w_osm = float(pcfg.get("weight_osm", 0.5))
                base = base + w_osm * _normalize(r_osm, cfg)
            else:
                logger.info("OSM waste GPKG has no usable geometries in %s", p)
        else:
            logger.info("osm_waste_gpkg not found (%s); solid proxy is CORINE-only.", p)
    out = _normalize(base, cfg).astype(np.float32)
    return np.maximum(out, 0.0)


def build_solid_waste_context_masks(
    cfg: dict[str, Any], ref: dict[str, Any]
) -> dict[str, np.ndarray]:
    """
    Binary masks on ``ref`` for visualization: CORINE 132/121 and one layer per OSM ``waste_family``.

    Keys: ``corine_clc_132``, ``corine_clc_121``, ``osm_<waste_family>``.
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("solid") or {}
    corine_path = Path(ref["corine_path"])
    if not corine_path.is_file():
        corine_path = resolve_path(root, Path(paths["corine"]))
    clc = warp_raster_to_ref(corine_path, ref, resampling=Resampling.nearest)
    # Match ``build_proxy_solid``: CORINE GeoTIFFs may store CLC L3 codes or EEA44 indices.
    clc_f = np.asarray(clc, dtype=np.float64)
    clc_int = np.full(clc_f.shape, -9999, dtype=np.int32)
    mfin = np.isfinite(clc_f)
    clc_int[mfin] = np.rint(clc_f[mfin]).astype(np.int32, copy=False)
    codes_132, _ = adapt_corine_classes_for_grid(clc_int, [132])
    codes_121, _ = adapt_corine_classes_for_grid(clc_int, [121])
    m132 = np.isin(clc_int, np.asarray(codes_132, dtype=np.int32)).astype(np.float32)
    m121 = np.isin(clc_int, np.asarray(codes_121, dtype=np.int32)).astype(np.float32)
    out: dict[str, np.ndarray] = {
        "corine_clc_132": m132,
        "corine_clc_121": m121,
    }
    osm_path = paths.get("osm_waste_gpkg")
    if not osm_path:
        return out
    p = Path(osm_path)
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        return out
    buf_m = float(pcfg.get("osm_point_buffer_m", 50.0))
    g_poly, g_pts = _load_osm_waste_polygons_points(p)
    if "waste_family" in g_poly.columns:
        g_poly = g_poly[g_poly["waste_family"].astype(str) != "wastewater_plant"].copy()
    if "waste_family" in g_pts.columns:
        g_pts = g_pts[g_pts["waste_family"].astype(str) != "wastewater_plant"].copy()
    crs = rasterio.crs.CRS.from_string(ref["crs"])
    _all_geoms, by_fam = _osm_geometries_for_solid(
        g_poly, g_pts, crs, point_buffer_m=buf_m
    )
    for fam in sorted(by_fam.keys()):
        geoms = by_fam[fam]
        out[f"osm_{fam}"] = _rasterize_mask(geoms, ref, burn=1.0).astype(np.float32)
    return out


def build_proxy_wastewater(cfg: dict[str, Any], ref: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Wastewater family proxy.

    - Agglomerations: dissolved union rasterized as binary presence.
    - Treatment plants: buffers in meters (EPSG:3035), union, rasterized;
      optional Gaussian smoothing (``scipy.ndimage``) if available.
    - Population and imperviousness: bilinear warps, normalized, weighted sum.

    Returns ``(proxy, imperv_valid_mask)`` where ``imperv_valid_mask`` is 1 where
    imperviousness source had finite data after warp.
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("wastewater") or {}
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    agg_path = resolve_path(root, Path(paths["uwwtd_agglomerations_gpkg"]))
    plant_path = resolve_path(root, Path(paths["uwwtd_plants_gpkg"]))
    pop_path = resolve_path(root, Path(paths["population_tif"]))
    imp_path = resolve_path(root, Path(paths["imperviousness"]))

    geoms_a: list = []
    if agg_path.is_file():
        ga = gpd.read_file(agg_path)
        if ga.crs:
            ga = ga.to_crs(crs)
            u = unary_union(list(ga.geometry))
            if not u.is_empty:
                geoms_a = [u]
    r_agg = _rasterize_mask(geoms_a, ref, all_touched=False)

    r_plant = np.zeros((int(ref["height"]), int(ref["width"])), dtype=np.float64)
    buf_m = float(pcfg.get("wwtp_buffer_m", 800.0))
    if plant_path.is_file():
        gp = gpd.read_file(plant_path)
        if gp.crs:
            gp = gp.to_crs(crs)
            geoms = [geom.buffer(buf_m) for geom in gp.geometry if geom is not None]
            if geoms:
                u = unary_union(geoms)
                r_plant = _rasterize_mask([u], ref, all_touched=False)
    sig = float(pcfg.get("wwtp_kernel_sigma_px", 2.0))
    if _HAS_SCIPY and sig > 0 and r_plant.max() > 0:
        r_plant = _ndimage.gaussian_filter(r_plant, sigma=sig, mode="nearest")

    pop = warp_raster_to_ref(pop_path, ref, resampling=Resampling.bilinear)
    imp_raw = _read_raster_path_or_dir(imp_path, ref, resampling=Resampling.bilinear)
    imperv_valid = np.isfinite(imp_raw) & (imp_raw > 0)
    imp = np.where(np.isfinite(imp_raw), imp_raw, 0.0)

    corine_path = Path(ref["corine_path"])
    clc = warp_raster_to_ref(corine_path, ref, resampling=Resampling.nearest)
    ind_codes = set(int(x) for x in (pcfg.get("industrial_clc_codes") or [131, 132, 133]))
    ind_mask = np.isin(clc, list(ind_codes)).astype(np.float64)

    n_agg = _normalize(r_agg, cfg)
    n_pl = _normalize(r_plant, cfg)
    n_pop = _normalize(pop, cfg)
    n_imp = _normalize(imp, cfg)
    n_ind = _normalize(ind_mask, cfg)

    wa = float(pcfg.get("weight_agglo_cover", 0.45))
    wp = float(pcfg.get("weight_wwtp", 0.35))
    wpop = float(pcfg.get("weight_pop", 0.35))
    wimp = float(pcfg.get("weight_imperv", 0.25))
    wind = float(pcfg.get("weight_industrial_clc", 0.08))

    comb = wa * n_agg + wp * n_pl + wpop * n_pop + wimp * n_imp + wind * n_ind
    out = _normalize(comb, cfg).astype(np.float32)
    return np.maximum(out, 0.0), imperv_valid.astype(np.uint8)


def build_wastewater_context_grids(cfg: dict[str, Any], ref: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Return intermediate rasters for wastewater proxy visualization on ``ref``.

    Keys (float32 unless noted):
    - ``uwwtd_agglomerations``: binary rasterized union (0/1)
    - ``uwwtd_treatment_plants``: buffered plants (optionally smoothed) (0..1+)
    - ``population``: warped population (raw values)
    - ``imperviousness``: warped imperviousness (raw values, NaN->0)
    - ``industrial_clc_mask``: CORINE industrial codes mask (0/1)
    - ``imperv_valid_mask``: uint8 mask (1 where imperviousness was finite and >0)
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("wastewater") or {}
    crs = rasterio.crs.CRS.from_string(ref["crs"])

    agg_path = resolve_path(root, Path(paths["uwwtd_agglomerations_gpkg"]))
    plant_path = resolve_path(root, Path(paths["uwwtd_plants_gpkg"]))
    pop_path = resolve_path(root, Path(paths["population_tif"]))
    imp_path = resolve_path(root, Path(paths["imperviousness"]))

    geoms_a: list = []
    if agg_path.is_file():
        ga = gpd.read_file(agg_path)
        if ga.crs:
            ga = ga.to_crs(crs)
            u = unary_union(list(ga.geometry))
            if not u.is_empty:
                geoms_a = [u]
    r_agg = _rasterize_mask(geoms_a, ref, all_touched=False).astype(np.float32)

    r_plant = np.zeros((int(ref["height"]), int(ref["width"])), dtype=np.float64)
    buf_m = float(pcfg.get("wwtp_buffer_m", 800.0))
    if plant_path.is_file():
        gp = gpd.read_file(plant_path)
        if gp.crs:
            gp = gp.to_crs(crs)
            geoms = [geom.buffer(buf_m) for geom in gp.geometry if geom is not None]
            if geoms:
                u = unary_union(geoms)
                r_plant = _rasterize_mask([u], ref, all_touched=False)
    sig = float(pcfg.get("wwtp_kernel_sigma_px", 2.0))
    if _HAS_SCIPY and sig > 0 and float(np.nanmax(r_plant)) > 0:
        r_plant = _ndimage.gaussian_filter(r_plant, sigma=sig, mode="nearest")
    r_plant = r_plant.astype(np.float32, copy=False)

    pop = warp_raster_to_ref(pop_path, ref, resampling=Resampling.bilinear).astype(np.float32, copy=False)

    imp_raw = _read_raster_path_or_dir(imp_path, ref, resampling=Resampling.bilinear)
    imperv_valid = (np.isfinite(imp_raw) & (imp_raw > 0)).astype(np.uint8)
    imp = np.where(np.isfinite(imp_raw), imp_raw, 0.0).astype(np.float32, copy=False)

    corine_path = Path(ref["corine_path"])
    clc = warp_raster_to_ref(corine_path, ref, resampling=Resampling.nearest)
    ind_codes = set(int(x) for x in (pcfg.get("industrial_clc_codes") or [131, 132, 133]))
    ind_mask = np.isin(clc, list(ind_codes)).astype(np.float32)

    return {
        "uwwtd_agglomerations": r_agg,
        "uwwtd_treatment_plants": r_plant,
        "population": pop,
        "imperviousness": imp,
        "industrial_clc_mask": ind_mask,
        "imperv_valid_mask": imperv_valid,
    }


def build_proxy_residual(cfg: dict[str, Any], ref: dict[str, Any]) -> np.ndarray:
    """
    Residual diffuse proxy: population, rural SMOD emphasis, imperviousness support.

    SMOD rural codes: configurable list (USER MUST CHECK against GHSL legend).
    """
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("residual") or {}
    pop_path = resolve_path(root, Path(paths["population_tif"]))
    smod_path = resolve_path(root, Path(paths["ghsl_smod_tif"]))
    imp_path = resolve_path(root, Path(paths["imperviousness"]))

    pop = warp_raster_to_ref(pop_path, ref, resampling=Resampling.bilinear)
    smod = warp_raster_to_ref(smod_path, ref, resampling=Resampling.nearest)
    imp = _read_raster_path_or_dir(imp_path, ref, resampling=Resampling.bilinear)
    imp = np.where(np.isfinite(imp), imp, 0.0)

    rural_codes = [int(x) for x in (pcfg.get("smod_rural_codes") or [11, 12, 13, 21, 22])]
    rural = np.isin(smod, rural_codes).astype(np.float64)
    settle = np.clip(imp / max(float(np.nanpercentile(imp, 99)) if np.any(imp > 0) else 1.0, 1e-6), 0.0, 1.0)

    wpop = float(pcfg.get("weight_pop", 0.5))
    wrur = float(pcfg.get("weight_rural_smod", 0.35))
    wimp = float(pcfg.get("weight_imperv", 0.15))

    comb = wpop * _normalize(pop, cfg) + wrur * _normalize(rural, cfg) + wimp * _normalize(settle, cfg)
    out = _normalize(comb, cfg).astype(np.float32)
    return np.maximum(out, 0.0)


def build_residual_context_grids(
    cfg: dict[str, Any], ref: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Raw / intermediate grids for J_Waste residual proxy (visualization)."""
    root = cfg["_project_root"]
    paths = cfg["paths"]
    pcfg = (cfg.get("proxy") or {}).get("residual") or {}
    pop_path = resolve_path(root, Path(paths["population_tif"]))
    smod_path = resolve_path(root, Path(paths["ghsl_smod_tif"]))
    imp_path = resolve_path(root, Path(paths["imperviousness"]))

    pop = warp_raster_to_ref(pop_path, ref, resampling=Resampling.bilinear).astype(
        np.float32, copy=False
    )
    smod = warp_raster_to_ref(smod_path, ref, resampling=Resampling.nearest)
    imp = _read_raster_path_or_dir(imp_path, ref, resampling=Resampling.bilinear)
    imp = np.where(np.isfinite(imp), imp, 0.0).astype(np.float32, copy=False)

    rural_codes = [int(x) for x in (pcfg.get("smod_rural_codes") or [11, 12, 13, 21, 22])]
    rural = np.isin(smod, rural_codes).astype(np.float32)
    hi = float(np.nanpercentile(imp, 99)) if np.any(imp > 0) else 1.0
    settle = np.clip(imp / max(hi, 1e-6), 0.0, 1.0).astype(np.float32, copy=False)
    return {
        "residual_pop": pop,
        "residual_ghsl_rural_mask": rural,
        "residual_imperv_01": settle,
    }


def build_all_proxies(cfg: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Run solid, wastewater, residual builders; return arrays + imperv_valid mask."""
    logger.info("Building proxy: solid")
    p_solid = build_proxy_solid(cfg, ref)
    logger.info("Building proxy: wastewater")
    p_ww, imperv_valid = build_proxy_wastewater(cfg, ref)
    logger.info("Building proxy: residual")
    p_res = build_proxy_residual(cfg, ref)
    return {
        "proxy_solid": p_solid,
        "proxy_wastewater": p_ww,
        "proxy_residual": p_res,
        "imperv_valid_mask": imperv_valid,
    }
