"""
Spatial proxies for **GNFR group-style** sectors (fugitive, industry, offroad, shipping, …).

Pipeline shape (see ``build_all_group_pg``):
  OSM geometries + YAML rules → **coverage** [0,1] per pixel
  CORINE L2 or CLC44 raster + YAML classes/weights → **land score** [0,1]
  z-score each → blend into **p_sector**; mix with population **p_pop** → **p_g**
  (when OSM+CLC are both near zero, use population-only fallback and flag it).

All arrays align to the same reference grid ``ref`` (height, width, affine, crs).
"""

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


def _buffer_osm_geom(geom: Any, point_m: float, line_m: float, polygon_m: float) -> Any:
    """Expand OSM geometries by type for coverage rasterization (distances in projected CRS, metres)."""
    if geom is None or geom.is_empty:
        return geom
    gt = geom.geom_type
    if gt == "Point":
        return geom.buffer(float(point_m)) if point_m > 0 else geom
    if gt == "MultiPoint":
        if point_m <= 0:
            return geom
        return unary_union([p.buffer(float(point_m)) for p in geom.geoms])
    if gt in ("LineString", "LinearRing"):
        return geom.buffer(float(line_m)) if line_m > 0 else geom
    if gt == "MultiLineString":
        return geom.buffer(float(line_m)) if line_m > 0 else geom
    if gt in ("Polygon", "MultiPolygon"):
        return geom.buffer(float(polygon_m)) if polygon_m > 0 else geom
    if gt == "GeometryCollection":
        parts = [
            _buffer_osm_geom(g, point_m, line_m, polygon_m)
            for g in geom.geoms
        ]
        parts = [p for p in parts if p is not None and not p.is_empty]
        if not parts:
            return geom
        return unary_union(parts)
    return geom


def _apply_osm_metric_buffers(
    gdf: gpd.GeoDataFrame,
    target_crs: Any,
    *,
    point_m: float,
    line_m: float,
    polygon_m: float,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if point_m <= 0 and line_m <= 0 and polygon_m <= 0:
        return gdf
    g = gdf.to_crs(target_crs)
    out = g.copy()
    out.geometry = g.geometry.apply(
        lambda gg: _buffer_osm_geom(gg, point_m, line_m, polygon_m)
    )
    return out


def pixel_area_m2(transform: rasterio.Affine) -> float:
    """Ground area (m²) of one raster cell for a north-up axis-aligned affine (|a·e| when b=d=0)."""
    return abs(float(transform[0]) * float(transform[4]))


def z_score(arr: np.ndarray) -> np.ndarray:
    """
    Map a raw raster to a **comparable 0–1 score**: robust to outliers, then linearly
    stretched between 1st and 99th percentiles; NaN/inf become 0.

    Not a statistical z-score (mean/variance); name is legacy. Output is suitable
    for blending weights with other 0–1 proxies.

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
    # Percentiles on >5M pixels would allocate a huge finite mask; subsample deterministically.
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


def mask_no_sector_z_signal_in_cams_cells(
    sector_z: np.ndarray,
    cam_cell_id: np.ndarray,
    *,
    max_abs_floor: float = 1e-10,
) -> np.ndarray:
    """
    Pixels that fall in a CAMS cell where the **peak** blended sector z-score (post
    :func:`z_score`, i.e. nonnegative ``p_sector`` or fugitive ``acc_score``) is at
    or below ``max_abs_floor``.

    When every pixel in a cell has ~0 sector score, spatial structure comes only from
    population via this fallback (same idea as per-pixel ``sector_z == 0``, but
    consistent for the whole CAMS cell).
    """
    z = np.asarray(sector_z, dtype=np.float64, order="C")
    cid = np.asarray(cam_cell_id, dtype=np.int64, order="C")
    if z.shape != cid.shape:
        raise ValueError(f"sector_z shape {z.shape} != cam_cell_id shape {cid.shape}")
    h, w = z.shape
    flat_z = z.ravel()
    flat_c = cid.ravel()
    valid = flat_c >= 0
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool)
    max_id = int(np.max(flat_c[valid])) + 1
    cell_peak = np.zeros(max_id, dtype=np.float64)
    np.maximum.at(cell_peak, flat_c[valid], flat_z[valid])
    peak_at_px = cell_peak[flat_c]
    out_flat = valid & (peak_at_px <= float(max_abs_floor))
    return out_flat.reshape(h, w)


# CORINE GeoTIFFs often store pan-European **CLC 1–44** legend indices. YAML may list CLC **Level 2**
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
    133: (9,),
    211: (12,),
    212: (13,),
    213: (14,),
    221: (18,),
    222: (16,),
    223: (17,),
    241: (20,),
    242: (21,),
    243: (22,),
}


def adapt_corine_classes_for_grid(clc_nn: np.ndarray, yaml_classes: list[int]) -> tuple[list[int], bool]:
    """
    Harmonise YAML CORINE **Level-2** codes (e.g. 121) with the raster’s **CLC 1–44** legend.

    If ``clc_nn`` looks like CLC 1–44 (max valid code ≤ 99), map Level-2 YAML codes to
    the underlying CLC44 class ids used in the GeoTIFF. If the raster already uses
    coarse / different coding (max > 99), return YAML classes unchanged.

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
    """Mean of binary **is this pixel one of the group’s CLC classes?** indicators (0–1)."""
    if not classes:
        return np.zeros(clc.shape, dtype=np.float32)
    ci = np.asarray(clc, dtype=np.int32, order="C")
    acc = np.zeros(clc.shape, dtype=np.float32)
    for c in classes:
        acc += (ci == int(c)).astype(np.float32, copy=False)
    inv = np.float32(1.0 / float(len(classes)))
    np.multiply(acc, inv, out=acc)
    return acc


def clc_raw_group_score_robust(clc: np.ndarray, yaml_classes: list[int]) -> np.ndarray:
    """Same contract as :func:`clc_raw_group_score`, but chooses CLC44 vs Level‑2 coding reliably.

    EU CORINE GeoTIFFs use either **CLC 1–44** indices (``max`` typically ≤ 48) or **Level‑2
    style codes** (e.g. 111–133, ``max`` often > 99). :func:`adapt_corine_classes_for_grid` uses
    ``max(clc)`` only; a handful of mis‑tagged / nodata pixels can push ``max`` above 99 and
    force literal matching on a CLC44 grid → **all‑zero CLC scores** (OSM still looks fine).

    We use the **99.5th percentile** of valid codes to decide encoding, then fall back to the
    other score if the primary branch is empty.
    """
    yaml_classes = [int(x) for x in yaml_classes]
    if not yaml_classes:
        return np.zeros(clc.shape, dtype=np.float32)
    ci = np.asarray(clc, dtype=np.int32, order="C")
    valid = ci != -9999
    if not np.any(valid):
        return np.zeros(clc.shape, dtype=np.float32)
    vals = ci[valid].astype(np.float64, copy=False)
    mx = float(np.max(vals))
    p995 = float(np.percentile(vals, 99.5))
    clc_use, _remapped = adapt_corine_classes_for_grid(ci, yaml_classes)
    s_map = clc_raw_group_score(ci, clc_use)
    s_raw = clc_raw_group_score(ci, yaml_classes)
    # Strong hint of CLC44 legend: almost all pixels are small integer class ids.
    looks_clc44 = p995 <= 48.0 and mx <= 96.0
    if looks_clc44:
        if float(np.nanmax(s_map)) > 1e-12:
            return s_map
        return s_raw
    if float(np.nanmax(s_raw)) > 1e-12:
        return s_raw
    return s_map


def clc_weighted_class_score(
    clc: np.ndarray, class_weights: dict[int, float]
) -> np.ndarray:
    """
    Like ``clc_raw_group_score`` but each CLC class contributes a **weight**; the
    result is a weighted average of per-class 0–1 scores (weights normalised to 1).

    If the raster uses CLC 1–44 indices (``max`` ≤ 99), each key is taken as
    a Level-2 code and passed through ``adapt_corine_classes_for_grid`` so 131/133
    map correctly. If ``max`` > 99, keys are raw CLC class values (exact match
    on ``-9999``-masked valid pixels only).
    """
    if not class_weights:
        return np.zeros(clc.shape, dtype=np.float32)
    wsum = float(sum(float(v) for v in class_weights.values()))
    if wsum <= 0:
        return np.zeros(clc.shape, dtype=np.float32)
    ci = np.asarray(clc, dtype=np.int32)
    valid = ci != -9999
    if not np.any(valid):
        return np.zeros(clc.shape, dtype=np.float32)
    mx = int(np.max(ci[valid]))
    acc = np.zeros(clc.shape, dtype=np.float32)
    if mx > 99:
        for c, w in class_weights.items():
            m = (ci == int(c)) & valid
            acc += (float(w) / wsum) * m.astype(np.float32, copy=False)
        return acc
    for c, w in class_weights.items():
        l2 = int(c)
        yaml = [l2]
        clc_use, _ = adapt_corine_classes_for_grid(clc, yaml)
        s = clc_raw_group_score(clc, clc_use)
        acc += (float(w) / wsum) * s
    return acc


def filter_osm_by_rules(gdf: gpd.GeoDataFrame, rules_block: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Keep OSM rows that match **any** of the attribute conjunctions in ``any_of``.

    ``rules_block`` is e.g. ``{any_of: [{landuse: quarry}, {natural: cliff}, ...]}``:
    each inner dict is AND over keys; the outer list is OR.
    """
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
    """
    Row indices of ``gdf`` that **may** intersect ``geom`` (spatial index pre-filter).

    GeoPandas ``sindex.query`` return shape varies by version (array vs ``(input_idx, tree_idx)`` tuple);
    fall back to a full ``intersects`` boolean scan if the index API is unavailable.
    """
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
    Approximate **fraction of each output pixel covered** by OSM geometries (0–1).

    Exact polygon–pixel intersection would be expensive; instead each ref pixel is
    split into ``subdivide_factor × subdivide_factor`` sub-cells, OSM is rasterised
    at that finer resolution, and the mean of sub-cell hits is the coverage fraction.

    Work proceeds in **spatial tiles** of ``tile_pixels`` so peak memory scales like
    ``(tile_pixels * subdivide_factor)²``, not full ``height × width``.
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
            # Slight buffer so geometries that only touch the tile edge are still picked up.
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
            # Finer grid inside this tile: rasterize once, then block-mean back to ref pixels.
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
    """Population count per cell → **density** (per m²) → same robust 0–1 scaling as ``z_score``."""
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
    *,
    group_order: tuple[str, ...] | None = None,
    cam_cell_id: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build **per CEIP group** proxy rasters used downstream with α weights.

    For each group id (``G1`` … or custom keys), returns a dict with:
      * ``osm_raw`` — OSM coverage fraction [0,1] (optional per-group ``osm_point_buffer_m`` /
        ``osm_line_buffer_m`` / ``osm_polygon_buffer_m`` in metres, projected CRS, applied before rasterize)
      * ``clc_raw`` — CORINE-based score [0,1]
      * ``p_sector`` — ``z_score(osm_raw)`` and ``z_score(clc_raw)`` blended with ``w_osm`` / ``w_clc``
      * ``p_g`` — final “group activity” proxy: sector blend vs population-only fallback
      * ``used_pop_fallback`` — 1 where OSM+CLC were both ~0 so ``p_g == p_pop``, or
        (when ``cam_cell_id`` is passed) where the CAMS cell’s peak sector z-score is ~0

    ``group_order`` fixes which ``group_specs["groups"]`` entries are built and in
    what order (must match CEIP alpha axis). If ``None``, uses dict insertion order
    of ``groups`` (YAML load order).
    """
    from PROXY.core.alpha import DEFAULT_GNFR_GROUP_ORDER

    # How much of p_g comes from (scaled sector proxy) vs population when sector signal exists.
    w_s = float(pcfg.get("w_sector_in_p", 0.8))
    w_p = float(pcfg.get("w_pop_in_p", 0.2))
    subf = int(pcfg.get("osm_subdivide_factor", 4))
    tile_px = int(pcfg.get("osm_tile_pixels", 256))
    # Default split inside “sector”: OSM coverage vs CORINE score (per-group YAML can override).
    w_osm_def = float(pcfg.get("w_osm_in_sector", 0.6))
    w_clc_def = float(pcfg.get("w_clc_in_sector", 0.4))

    out: dict[str, dict[str, np.ndarray]] = {}
    groups = {str(k): v for k, v in dict(group_specs.get("groups") or {}).items()}
    order = group_order if group_order is not None else tuple(groups.keys()) or DEFAULT_GNFR_GROUP_ORDER
    for gid in order:
        spec = groups.get(str(gid))
        if spec is None:
            raise KeyError(
                f"build_all_group_pg: group {gid!r} not in group_specs (have {sorted(groups)})"
            )
        # Group-level override for population blend weights (optional).
        #
        # If the group provides exactly one of (w_sector_in_p, w_pop_in_p), derive the
        # other as 1-x so that the blend stays interpretable.
        # If the group provides both, normalize to sum 1 when possible.
        ws_raw = spec.get("w_sector_in_p", None)
        wp_raw = spec.get("w_pop_in_p", None)
        if ws_raw is None and wp_raw is None:
            w_s_g, w_p_g = w_s, w_p
        elif ws_raw is None and wp_raw is not None:
            w_p_g = float(wp_raw)
            w_s_g = 1.0 - w_p_g
        elif ws_raw is not None and wp_raw is None:
            w_s_g = float(ws_raw)
            w_p_g = 1.0 - w_s_g
        else:
            w_s_g = float(ws_raw)
            w_p_g = float(wp_raw)
            s = w_s_g + w_p_g
            if abs(s) > 1e-12:
                w_s_g /= s
                w_p_g /= s
        w_osm = float(spec.get("w_osm_in_sector", w_osm_def))
        w_clc = float(spec.get("w_clc_in_sector", w_clc_def))
        rules = spec.get("osm_rules") or {}
        gsub = filter_osm_by_rules(osm_gdf, rules)
        buf_p = float(spec.get("osm_point_buffer_m", 0) or 0)
        buf_l = float(spec.get("osm_line_buffer_m", 0) or 0)
        buf_poly = float(spec.get("osm_polygon_buffer_m", 0) or 0)
        if buf_p > 0 or buf_l > 0 or buf_poly > 0:
            tgt = rasterio.crs.CRS.from_string(ref["crs"])
            gsub = _apply_osm_metric_buffers(
                gsub,
                tgt,
                point_m=buf_p,
                line_m=buf_l,
                polygon_m=buf_poly,
            )
        osm_raw = osm_coverage_fraction(gsub, ref, subdivide_factor=subf, tile_pixels=tile_px)
        cw = spec.get("corine_class_weights")
        if cw:
            cwmap = {int(k): float(v) for k, v in dict(cw).items()}
            clc_raw = clc_weighted_class_score(clc, cwmap)
        else:
            yaml_clc = [int(x) for x in (spec.get("corine_classes") or [])]
            clc_raw = clc_raw_group_score_robust(clc, yaml_clc)
        # Put OSM and CLC on the same scale, then linear mix → single “sector structure” score.
        z_o = z_score(osm_raw)
        z_c = z_score(clc_raw)
        p_sector = (w_osm * z_o + w_clc * z_c).astype(np.float32)
        # No OSM and no CLC signal: trust population layout only (avoids diluting tiny signals).
        raw_sum = osm_raw + clc_raw
        use_pop = raw_sum < 1e-12
        if cam_cell_id is not None:
            z_floor = float(pcfg.get("cell_no_sector_z_floor", 1e-10))
            use_pop = use_pop | mask_no_sector_z_signal_in_cams_cells(
                p_sector, cam_cell_id, max_abs_floor=z_floor
            )
        p_g = np.where(
            use_pop,
            p_pop,
            (w_s_g * p_sector + w_p_g * p_pop).astype(np.float32),
        ).astype(np.float32)
        
        out[str(gid)] = {
            "osm_raw": osm_raw,
            "clc_raw": clc_raw.astype(np.float32),
            "p_sector": p_sector,
            "p_g": p_g,
            "used_pop_fallback": use_pop.astype(np.uint8),
        }
    return out
