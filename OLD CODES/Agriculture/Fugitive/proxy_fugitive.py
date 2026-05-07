"""Spatial proxies for fugitive groups: OSM coverage, CORINE support, population blend."""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio import windows as rio_windows
from rasterio.enums import MergeAlg
from shapely.geometry import box, mapping
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def pixel_area_m2(transform: rasterio.Affine) -> float:
    """Pixel area for axis-aligned affine (|a*e| when b=d=0)."""
    return abs(float(transform[0]) * float(transform[4]))


def z_score(arr: np.ndarray) -> np.ndarray:
    """
    Clip to 1st–99th percentile of finite values, rescale to [0,1], NaN/inf -> 0.

    Uses float32 and in-place ops to avoid float64 temporaries the size of the
    full raster. For very large grids, percentiles use a fixed random subsample
    so we never build a full ``x[isfinite]`` copy.
    """
    x = np.asarray(arr, dtype=np.float32, order="C")
    out = np.zeros_like(x, dtype=np.float32)
    mask = np.isfinite(x)
    if not np.any(mask):
        return out

    flat = x.ravel()
    n = int(flat.size)
    if n > 5_000_000:
        rng = np.random.default_rng(0)
        sub = flat[rng.integers(0, n, size=2_000_000, endpoint=False)]
        sub = sub[np.isfinite(sub)]
    else:
        sub = flat[mask.ravel()]
    if sub.size == 0:
        return out
    lo = float(np.quantile(sub, 0.01))
    hi = float(np.quantile(sub, 0.99))
    if hi <= lo:
        hi = lo + 1e-12
    scale = np.float32(1.0 / (hi - lo))
    lo32 = np.float32(lo)

    out.fill(0.0)
    np.subtract(x, lo32, out=out, where=mask)
    np.multiply(out, scale, out=out, where=mask)
    np.clip(out, 0.0, 1.0, out=out)
    return out


# CORINE GeoTIFFs in this repo often store the pan-European **CLC 1–44** legend (see
# ``urbem_interface/config/factory_bundled/corine_classes.json``). YAML lists CLC **Level 2**
# codes (e.g. 121). Map only the codes used in fugitive groups; extend if you add more.
_CLC_LEVEL2_TO_CLC44: dict[int, tuple[int, ...]] = {
    111: (1,),
    112: (2,),
    121: (3,),
    122: (4,),
    123: (5,),
    124: (6,),
    131: (7, 8, 9),
    132: (7, 8, 9),
    133: (9,),  # Construction sites / mineral extraction overlap handled like 131 in some tiles
    # Agriculture (CLC Level 2 → Level 3 codes 12–22 in pan-European 1–44 legend)
    211: (12,),  # Non-irrigated arable land
    212: (13,),  # Permanently irrigated land
    213: (14,),  # Rice fields
    221: (18,),  # Vineyards
    222: (16,),  # Fruit trees and berry plantations
    223: (17,),  # Olive groves
    241: (20,),  # Annual crops associated with permanent crops
    242: (21,),  # Complex cultivation patterns
    243: (22,),  # Land principally occupied by agriculture
}


def adapt_corine_classes_for_grid(clc_nn: np.ndarray, yaml_classes: list[int]) -> tuple[list[int], bool]:
    """
    If ``clc_nn`` looks like CLC 1–44 (max code ≤ 99), map Level-2 YAML codes to CLC44 indices.

    Returns ``(classes_for_scoring, remapped)``.
    """
    base = [int(x) for x in yaml_classes]
    if not base:
        return [], False
    valid = clc_nn != -9999
    if not np.any(valid):
        return base, False
    mx = int(np.max(clc_nn[valid]))
    if mx > 99:
        return base, False
    out: list[int] = []
    for ci in base:
        if ci in _CLC_LEVEL2_TO_CLC44:
            out.extend(_CLC_LEVEL2_TO_CLC44[ci])
        elif 1 <= ci <= 44:
            out.append(ci)
        else:
            logger.warning(
                "CORINE class %d has no Level-2→CLC1–44 mapping (raster max=%d); skipped for this group.",
                ci,
                mx,
            )
    out_u = sorted(set(out))
    if not out_u:
        return base, False
    return out_u, True


def clc_raw_group_score(clc: np.ndarray, classes: list[int]) -> np.ndarray:
    """Mean of binary class indicators (mutually exclusive CLC → typically 0 or 1)."""
    if not classes:
        return np.zeros(clc.shape, dtype=np.float32)
    ci = np.asarray(clc, dtype=np.int32, order="C")
    acc = np.zeros(clc.shape, dtype=np.float32)
    for c in classes:
        acc += (ci == int(c)).astype(np.float32, copy=False)
    inv = np.float32(1.0 / float(len(classes)))
    np.multiply(acc, inv, out=acc)
    return acc


def filter_osm_by_rules(gdf: gpd.GeoDataFrame, rules_block: dict[str, Any]) -> gpd.GeoDataFrame:
    """``rules_block`` is e.g. ``{any_of: [{landuse: quarry}, ...]}``."""
    rules = (rules_block or {}).get("any_of") or []
    if gdf.empty or not rules:
        return gdf.iloc[0:0].copy()

    def row_ok(row: Any) -> bool:
        for rule in rules:
            ok = True
            for k, v in rule.items():
                val = row.get(k)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    s = ""
                else:
                    s = str(val).strip().lower()
                if s != str(v).strip().lower():
                    ok = False
                    break
            if ok:
                return True
        return False

    mask = gdf.apply(row_ok, axis=1)
    return gdf.loc[mask].copy()


def _sindex_intersects(gdf: gpd.GeoDataFrame, geom) -> np.ndarray:
    """Return row indices into ``gdf`` that may intersect ``geom`` (best-effort across GeoPandas versions)."""
    try:
        raw = gdf.sindex.query(geom, predicate="intersects")
    except (TypeError, AttributeError, ValueError):
        return np.flatnonzero(gdf.intersects(geom).to_numpy()).astype(np.intp, copy=False)
    if isinstance(raw, tuple):
        if len(raw) != 2:
            return np.asarray([], dtype=np.intp)
        _in, tree_idx = raw
        if np.asarray(_in).size == 0:
            return np.asarray([], dtype=np.intp)
        return np.asarray(tree_idx, dtype=np.intp, copy=False)
    return np.asarray(raw, dtype=np.intp, copy=False)


def osm_coverage_fraction(
    gdf: gpd.GeoDataFrame,
    ref: dict[str, Any],
    *,
    subdivide_factor: int = 4,
    tile_pixels: int = 256,
) -> np.ndarray:
    """
    Approximate ``area(OSM ∩ pixel) / area(pixel)`` by sub-sampling each fine pixel
    on a ``subdivide_factor`` grid and rasterizing OSM at that resolution.

    Rasterization is done in **spatial tiles** so peak memory stays on the order
    of ``(tile_pixels * subdivide_factor)^2`` instead of the full grid size.
    """
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    crs = rasterio.crs.CRS.from_string(ref["crs"])
    out = np.zeros((h, w), dtype=np.float32)
    if gdf.empty:
        return out
    g3035 = gdf.to_crs(crs)
    geom_ok = g3035.geometry.notna() & (~g3035.geometry.is_empty)
    if not bool(np.any(geom_ok)):
        return out
    f = max(1, int(subdivide_factor))
    tp = max(32, int(tile_pixels))

    n_tiles = int(np.ceil(h / tp) * np.ceil(w / tp))
    if n_tiles > 1:
        logger.info(
            "OSM coverage: tiling %d x %d grid into ~%d tiles (tile=%d px, subdiv=%d)",
            h,
            w,
            n_tiles,
            tp,
            f,
        )

    for r0 in range(0, h, tp):
        r1 = min(h, r0 + tp)
        h_t = r1 - r0
        for c0 in range(0, w, tp):
            c1 = min(w, c0 + tp)
            w_t = c1 - c0
            win = rio_windows.Window(c0, r0, w_t, h_t)
            left, bottom, right, top = rio_windows.bounds(win, transform)
            tile_box = box(left, bottom, right, top)
            buf = max(abs(float(transform[0])), abs(float(transform[4]))) * 2.0
            qbox = tile_box.buffer(buf)
            idx = _sindex_intersects(g3035, qbox)
            if idx.size == 0:
                continue
            sub_g = g3035.iloc[np.unique(idx)]
            parts = []
            for g in sub_g.geometry:
                if g is None or g.is_empty:
                    continue
                gi = g.intersection(tile_box)
                if not gi.is_empty:
                    parts.append(gi)
            if not parts:
                continue
            u = unary_union(parts)
            if u.is_empty:
                continue
            h2 = h_t * f
            w2 = w_t * f
            t2 = rasterio.transform.from_bounds(left, bottom, right, top, width=w2, height=h2)
            sub = features.rasterize(
                [(mapping(u), 1)],
                out_shape=(h2, w2),
                transform=t2,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
                merge_alg=MergeAlg.replace,
            )
            sub = sub.reshape(h_t, f, w_t, f).astype(np.float32)
            cov = np.mean(sub, axis=(1, 3))
            out[r0:r1, c0:c1] = np.clip(cov, 0.0, 1.0).astype(np.float32)
    return out


def build_p_pop(pop: np.ndarray, ref: dict[str, Any]) -> np.ndarray:
    """Population density proxy then z-scored."""
    area = pixel_area_m2(ref["transform"])
    dens = np.where(np.isfinite(pop) & (pop >= 0), pop.astype(np.float64) / max(area, 1e-6), np.nan)
    return z_score(dens)


def build_all_group_pg(
    clc: np.ndarray,
    osm_gdf: gpd.GeoDataFrame,
    group_specs: dict[str, Any],
    ref: dict[str, Any],
    pcfg: dict[str, Any],
    p_pop: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Return per group id ``G1``..``G4`` dict with keys:
    ``osm_raw``, ``clc_raw``, ``p_sector``, ``p_g``, ``used_pop_fallback`` (uint8 raster).
    """
    w_osm = float(pcfg.get("w_osm_in_sector", 0.6))
    w_clc = float(pcfg.get("w_clc_in_sector", 0.4))
    w_s = float(pcfg.get("w_sector_in_p", 0.8))
    w_p = float(pcfg.get("w_pop_in_p", 0.2))
    subf = int(pcfg.get("osm_subdivide_factor", 4))
    tile_px = int(pcfg.get("osm_tile_pixels", 256))

    out: dict[str, dict[str, np.ndarray]] = {}
    groups = group_specs.get("groups") or {}
    for gid, spec in groups.items():
        if not str(gid).startswith("G"):
            continue
        rules = spec.get("osm_rules") or {}
        gsub = filter_osm_by_rules(osm_gdf, rules)
        osm_raw = osm_coverage_fraction(gsub, ref, subdivide_factor=subf, tile_pixels=tile_px)
        yaml_clc = [int(x) for x in (spec.get("corine_classes") or [])]
        clc_use, _ = adapt_corine_classes_for_grid(clc, yaml_clc)
        clc_raw = clc_raw_group_score(clc, clc_use)
        z_o = z_score(osm_raw)
        z_c = z_score(clc_raw)
        p_sector = (w_osm * z_o + w_clc * z_c).astype(np.float32)
        raw_sum = osm_raw + clc_raw
        use_pop = raw_sum < 1e-12
        p_g = np.where(
            use_pop,
            p_pop,
            (w_s * p_sector + w_p * p_pop).astype(np.float32),
        ).astype(np.float32)
        out[str(gid)] = {
            "osm_raw": osm_raw,
            "clc_raw": clc_raw.astype(np.float32),
            "p_sector": p_sector,
            "p_g": p_g,
            "used_pop_fallback": use_pop.astype(np.uint8),
        }
    return out
