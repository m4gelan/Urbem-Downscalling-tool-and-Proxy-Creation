"""OSM PBF: road length proxy and landuse/building area proxies on ref grid."""

from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.transform import rowcol
from shapely.geometry import mapping

from PROXY.core.dataloaders import resolve_path as project_resolve

logger = logging.getLogger(__name__)

# Expected layers/column from data/Solvents/build_osm_solvent_layers.py (or manual GPKG).
SOLVENT_POLY_LAYER = "osm_solvent_polygons"
SOLVENT_POINT_LAYER = "osm_solvent_points"

# solvent_family values -> what each OSM subgroup covers (for logs / map legends).
OSM_SOLVENT_FAMILY_COVER: dict[str, str] = {
    "solvent_industrial": "OSM landuse industrial / commercial-industrial (polygons)",
    "solvent_commercial": "OSM landuse commercial / retail (polygons)",
    "solvent_service": "OSM civic / public / service (polygons and buffered points)",
    "solvent_other": "Other solvent-related OSM features not classified above",
}

# Layer names from data/OSM/build_osm_solvent_layers.py (multi-layer GPKG).
OSM_SOLVENT_LAYER_COVER: dict[str, str] = {
    "osm_landuse": "landuse polygons (residential/commercial/industrial/retail/port subset)",
    "osm_buildings": "building polygons (residential/commercial/industrial subset)",
    "osm_aeroway": "aeroway polygons",
    "osm_roads": "highway line geometry (length sampled onto grid; infra archetype)",
}

def _highway_scale(s: Any, scales: dict[str, float]) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return float(scales.get("default_scale", 1.0))
    h = str(s).lower().strip()
    if h in ("motorway", "motorway_link"):
        return float(scales.get("motorway_scale", 3.0))
    if h in ("trunk", "trunk_link", "primary", "primary_link"):
        return float(scales.get("primary_scale", 2.0))
    if h in ("secondary", "secondary_link", "tertiary", "tertiary_link"):
        return float(scales.get("secondary_scale", 1.5))
    return float(scales.get("default_scale", 1.0))


def accumulate_road_length_grid(
    lines: gpd.GeoDataFrame,
    highway_col: str | None,
    ref: dict,
    scales: dict[str, float],
    *,
    sample_m: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (road_length, weighted_road_length) float32 (H,W) in metres-equivalent counts.

    Each line is sampled every `sample_m` metres in EPSG:3035; each sample adds
    (segment_length / n_samples) * weight to its pixel (approximate length allocation).
    """
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    acc = np.zeros((h, w), dtype=np.float64)
    wacc = np.zeros((h, w), dtype=np.float64)
    if lines.empty:
        return acc.astype(np.float32), wacc.astype(np.float32)

    g = lines.to_crs(ref["crs"])
    hc = highway_col if highway_col and highway_col in g.columns else None
    hws = g[hc] if hc else pd.Series([None] * len(g), index=g.index)
    for geom, hw in zip(g.geometry, hws, strict=False):
        if geom is None or geom.is_empty:
            continue
        L = float(geom.length)
        if L <= 0:
            continue
        n = max(2, int(np.ceil(L / float(sample_m))))
        wt = _highway_scale(hw, scales)
        pts = [geom.interpolate(i / (n - 1), normalized=True) for i in range(n)]
        contrib = L / float(n)
        for pt in pts:
            r, c = rowcol(transform, float(pt.x), float(pt.y))
            if 0 <= r < h and 0 <= c < w:
                acc[r, c] += contrib
                wacc[r, c] += contrib * wt
    return acc.astype(np.float32), wacc.astype(np.float32)


def rasterize_polygon_sum(
    polys: gpd.GeoDataFrame,
    ref: dict,
    *,
    value_col: str | None = None,
) -> np.ndarray:
    """Sum rasterized polygon areas (value 1 per polygon, merge ADD) as overlap count proxy."""
    h, w = int(ref["height"]), int(ref["width"])
    transform = ref["transform"]
    if polys.empty:
        return np.zeros((h, w), dtype=np.float32)
    g = polys.to_crs(ref["crs"])
    shapes = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        v = float(row[value_col]) if value_col and value_col in g.columns else 1.0
        shapes.append((mapping(geom), v))
    if not shapes:
        return np.zeros((h, w), dtype=np.float32)
    out = features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        merge_alg=MergeAlg.add,
    )
    return out.astype(np.float32)


def load_solvent_polygons_points(gpkg_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read ``osm_solvent_polygons`` / ``osm_solvent_points`` or a single layer with ``solvent_family``."""
    try:
        import fiona
    except ImportError:
        fiona = None  # type: ignore[assignment]
    poly = gpd.GeoDataFrame()
    pts = gpd.GeoDataFrame()
    if fiona is not None:
        try:
            layers = fiona.listlayers(str(gpkg_path))
        except Exception:
            layers = []
        if SOLVENT_POLY_LAYER in layers:
            poly = gpd.read_file(gpkg_path, layer=SOLVENT_POLY_LAYER)
        if SOLVENT_POINT_LAYER in layers:
            pts = gpd.read_file(gpkg_path, layer=SOLVENT_POINT_LAYER)
    if poly.empty and pts.empty:
        g = gpd.read_file(gpkg_path)
        fam_col = "solvent_family"
        if g.empty or fam_col not in g.columns:
            return poly, pts
        gt = g.geometry.geom_type
        pts = g[gt == "Point"].copy()
        poly = g[~gt.isin(["Point", "MultiPoint"])].copy()
    return poly, pts


def _solvent_geometries_by_family(
    g_poly: gpd.GeoDataFrame,
    g_pts: gpd.GeoDataFrame,
    crs: rasterio.crs.CRS,
    *,
    point_buffer_m: float,
    family_col: str,
) -> dict[str, list]:
    by_fam: dict[str, list] = defaultdict(list)
    if not g_poly.empty:
        gp = g_poly
        if gp.crs is None:
            logger.warning(
                "OSM solvent polygons have no CRS; assuming EPSG:3035 (Solvent OSM GPKG convention)."
            )
            gp = gp.set_crs("EPSG:3035")
        gp = gp.to_crs(crs)
        for _, row in gp.iterrows():
            geom = row.geometry
            fam = str(row.get(family_col, "solvent_other") or "solvent_other")
            if geom is None or geom.is_empty:
                continue
            by_fam[fam].append(geom)
    if not g_pts.empty:
        gt = g_pts
        if gt.crs is None:
            logger.warning(
                "OSM solvent points have no CRS; assuming EPSG:3035 (Solvent OSM GPKG convention)."
            )
            gt = gt.set_crs("EPSG:3035")
        gt = gt.to_crs(crs)
        for _, row in gt.iterrows():
            geom = row.geometry
            fam = str(row.get(family_col, "solvent_other") or "solvent_other")
            if geom is None or geom.is_empty:
                continue
            b = geom.buffer(float(point_buffer_m))
            by_fam[fam].append(b)
    return dict(by_fam)


def _rasterize_geom_list(geoms: list, ref: dict) -> np.ndarray:
    if not geoms:
        h, w = int(ref["height"]), int(ref["width"])
        return np.zeros((h, w), dtype=np.float32)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=ref["crs"])
    return rasterize_polygon_sum(gdf, ref)


def _rasterize_solvent_gpkg_multi_layer(
    p: Path,
    layer_names: list[str],
    ref: dict,
    osm_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Layers from ``data/OSM/build_osm_solvent_layers.py``: polygons per layer + ``osm_roads`` as lines.

    Returns polygon rasters under layer names; road grids as ``_roads_rl`` / ``_roads_rw``.
    """
    out: dict[str, np.ndarray] = {}
    sample_m = float(osm_cfg.get("sample_m", 120.0))
    for name in layer_names:
        try:
            g = gpd.read_file(p, layer=name)
        except Exception as exc:
            logger.warning("Solvent GPKG: skip layer %r: %s", name, exc)
            continue
        if g.empty:
            continue
        if g.crs is None:
            logger.warning(
                "Solvent GPKG layer %r has no CRS; assuming EPSG:3035 (build_osm_solvent_layers convention).",
                name,
            )
            g = g.set_crs("EPSG:3035")
        g = g.to_crs(ref["crs"])
        gt = g.geometry.geom_type
        is_line = bool(gt.notna().all() and gt.isin(["LineString", "MultiLineString"]).all())
        if name == "osm_roads" or is_line:
            hw = "highway" if "highway" in g.columns else None
            rl, rw = accumulate_road_length_grid(g, hw, ref, osm_cfg, sample_m=sample_m)
            out["_roads_rl"] = rl
            out["_roads_rw"] = rw
            cover = OSM_SOLVENT_LAYER_COVER.get(name, name)
            logger.info("Solvent OSM GPKG: layer=%s (%s) — line features %d", name, cover, len(g))
        else:
            out[name] = rasterize_polygon_sum(g, ref)
            cover = OSM_SOLVENT_LAYER_COVER.get(name, name)
            logger.info("Solvent OSM GPKG: layer=%s (%s) — polygon features %d", name, cover, len(g))
    return out


def rasterize_solvent_gpkg_by_family(
    root: Path,
    cfg: dict[str, Any],
    ref: dict,
) -> dict[str, np.ndarray]:
    """
    Rasters from ``paths.osm_solvent_gpkg``:

    - **Multi-layer** (``osm.osm_solvent_layers`` non-empty): layers such as ``osm_landuse``, ``osm_buildings``,
      ``osm_aeroway`` (polygons) and ``osm_roads`` (lines -> ``_roads_rl`` / ``_roads_rw``).
    - **Family** mode (no ``osm_solvent_layers``): ``osm_solvent_polygons`` / ``osm_solvent_points`` with
      ``solvent_family`` column; binary mask per family.
    """
    if os.environ.get("SOLVENTS_SKIP_OSM", "").strip().lower() in ("1", "true", "yes"):
        return {}
    paths = cfg.get("paths") or {}
    raw = paths.get("osm_solvent_gpkg")
    if not raw:
        return {}
    p = project_resolve(root, Path(raw))
    if not p.is_file():
        return {}
    osm_cfg = cfg.get("osm") or {}
    layer_list = osm_cfg.get("osm_solvent_layers")
    if layer_list:
        return _rasterize_solvent_gpkg_multi_layer(p, list(layer_list), ref, osm_cfg)

    fam_col = str(osm_cfg.get("solvent_family_column", "solvent_family"))
    buf_m = float(osm_cfg.get("osm_point_buffer_m", 50.0))
    g_poly, g_pts = load_solvent_polygons_points(p)
    crs = rasterio.crs.CRS.from_string(str(ref["crs"]))
    by_fam = _solvent_geometries_by_family(
        g_poly, g_pts, crs, point_buffer_m=buf_m, family_col=fam_col
    )
    out: dict[str, np.ndarray] = {}
    for fam in sorted(by_fam.keys()):
        geoms = by_fam[fam]
        out[fam] = _rasterize_geom_list(geoms, ref)
        cover = OSM_SOLVENT_FAMILY_COVER.get(fam, fam)
        logger.info("Solvent OSM GPKG: solvent_family=%s (%s) — %d geometries", fam, cover, len(geoms))
    return out


def aggregate_solvent_families_to_osm_channels(
    family_masks: dict[str, np.ndarray],
    family_to_ch: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-family masks into ``service_osm`` and ``industry_osm`` sums (channel: serv|ind)."""
    if not family_masks:
        raise ValueError("aggregate_solvent_families_to_osm_channels: empty family_masks")
    first = next(iter(family_masks.values()))
    h, w = int(first.shape[0]), int(first.shape[1])
    serv = np.zeros((h, w), dtype=np.float32)
    ind = np.zeros((h, w), dtype=np.float32)
    for fam, mask in family_masks.items():
        ch = str(family_to_ch.get(fam, "ind")).strip().lower()
        m = np.asarray(mask, dtype=np.float32)
        if ch == "serv":
            serv += m
        elif ch == "ind":
            ind += m
    return serv, ind


def load_osm_indicators(
    root: Path,
    cfg: dict[str, Any],
    ref: dict,
) -> dict[str, np.ndarray]:
    """
    Build OSM-derived raw indicators on ref grid.

    Keys: service_osm, industry_osm, industry_buildings (combined proxy),
    transport_area (rough: industrial+commercial footprint here as placeholder),
    roof_area (zeros unless DSM available).

    When ``paths.osm_solvent_gpkg`` is set:

    - **Multi-layer** GPKG (``osm.osm_solvent_layers``): polygon layers are mapped with
      ``osm.solvent_layer_to_archetype`` (``serv`` / ``ind``); ``osm_roads`` supplies
      ``_road_length_raw`` / ``_weighted_road_raw`` instead of the roads PBF when present.
    - **Family** GPKG: ``osm_solvent_polygons`` / ``osm_solvent_points`` or a single layer with
      ``solvent_family``, using ``osm.solvent_family_to_archetype``.

    If the GPKG yields no usable polygon masks, landuse falls back to the landuse PBF.

    Set env ``SOLVENTS_SKIP_OSM=1`` to skip PBF reads (zeros; for dry runs).
    """
    paths = cfg["paths"]
    if os.environ.get("SOLVENTS_SKIP_OSM", "").strip().lower() in ("1", "true", "yes"):
        h, wgrid = int(ref["height"]), int(ref["width"])
        zeros = np.zeros((h, wgrid), dtype=np.float32)
        return {
            "service_osm": zeros.copy(),
            "industry_osm": zeros.copy(),
            "industry_buildings": zeros.copy(),
            "transport_area": zeros.copy(),
            "roof_area": zeros.copy(),
            "_road_length_raw": zeros.copy(),
            "_weighted_road_raw": zeros.copy(),
            "road_length": zeros.copy(),
            "weighted_road_length": zeros.copy(),
        }
    osm_cfg = cfg.get("osm") or {}
    w, s, e, n = ref["domain_bbox_wgs84"]
    pad = float(osm_cfg.get("bbox_pad_deg", 0.02))
    bbox = (w - pad, s - pad, e + pad, n + pad)
    h, wgrid = int(ref["height"]), int(ref["width"])
    zeros = np.zeros((h, wgrid), dtype=np.float32)
    out: dict[str, np.ndarray] = {
        "service_osm": zeros.copy(),
        "industry_osm": zeros.copy(),
        "industry_buildings": zeros.copy(),
        "transport_area": zeros.copy(),
        "roof_area": zeros.copy(),
    }

    gpkg_r = rasterize_solvent_gpkg_by_family(root, cfg, ref) or {}
    rl_g = gpkg_r.pop("_roads_rl", None)
    rw_g = gpkg_r.pop("_roads_rw", None)

    smap = dict(
        osm_cfg.get("solvent_layer_to_archetype")
        or osm_cfg.get("solvent_family_to_archetype")
        or {}
    )
    gpkg_poly_ok = False
    if gpkg_r and smap:
        poly_masks = {
            k: v
            for k, v in gpkg_r.items()
            if not str(k).startswith("_")
            and str(smap.get(k, "")).lower() in ("serv", "ind")
        }
        if poly_masks:
            zmax = [float(np.max(v)) for v in poly_masks.values()]
            if any(z > 0 for z in zmax):
                s_osm, i_osm = aggregate_solvent_families_to_osm_channels(poly_masks, smap)
                out["service_osm"] = s_osm
                out["industry_osm"] = i_osm
                out["industry_buildings"] = out["industry_osm"].copy()
                gpkg_poly_ok = True

    if not gpkg_poly_ok:
        land_pbf = project_resolve(root, Path(paths.get("osm_landuse_buildings_pbf", "")))
        if land_pbf.is_file():
            try:
                read_kw2: dict[str, Any] = {"bbox": bbox}
                mr2 = os.environ.get("SOLVENTS_OSM_MAX_ROWS_LANDUSE")
                if mr2:
                    read_kw2["rows"] = int(mr2)
                else:
                    mrl = osm_cfg.get("max_rows_landuse")
                    if mrl is not None:
                        read_kw2["rows"] = int(mrl)
                mp = gpd.read_file(
                    land_pbf,
                    layer=osm_cfg.get("landuse_layer", "multipolygons"),
                    **read_kw2,
                )
                capm = os.environ.get("SOLVENTS_OSM_LANDUSE_CAP")
                if capm:
                    mp = mp.iloc[: int(capm)]
                elif osm_cfg.get("landuse_feature_cap") is not None:
                    mp = mp.iloc[: int(osm_cfg["landuse_feature_cap"])]
                lu_ind = set(x.lower() for x in osm_cfg.get("landuse_industrial", []))
                lu_srv = set(x.lower() for x in osm_cfg.get("landuse_service", []))

                def landuse_series(gdf: gpd.GeoDataFrame) -> pd.Series:
                    if "landuse" in gdf.columns:
                        return gdf["landuse"].astype(str).str.lower()
                    if "fclass" in gdf.columns:
                        return gdf["fclass"].astype(str).str.lower()
                    return pd.Series("", index=gdf.index)

                lu = landuse_series(mp)
                ind_g = mp[lu.isin(lu_ind)]
                srv_g = mp[lu.isin(lu_srv)]
                out["industry_osm"] = rasterize_polygon_sum(ind_g, ref)
                out["service_osm"] = rasterize_polygon_sum(srv_g, ref)
                out["industry_buildings"] = out["industry_osm"].copy()
            except Exception as exc:
                warnings.warn(f"OSM landuse read failed: {exc}", stacklevel=1)

    if rl_g is not None and rw_g is not None:
        out["_road_length_raw"] = rl_g
        out["_weighted_road_raw"] = rw_g
    else:
        roads_pbf = project_resolve(root, Path(paths.get("osm_roads_pbf", "")))
        if roads_pbf.is_file():
            try:
                read_kw: dict[str, Any] = {"bbox": bbox}
                mr = os.environ.get("SOLVENTS_OSM_MAX_ROWS")
                if mr:
                    read_kw["rows"] = int(mr)
                else:
                    mrc = osm_cfg.get("max_rows_roads")
                    if mrc is not None:
                        read_kw["rows"] = int(mrc)
                lines = gpd.read_file(
                    roads_pbf,
                    layer=osm_cfg.get("roads_layer", "lines"),
                    **read_kw,
                )
                cap = os.environ.get("SOLVENTS_OSM_ROADS_CAP")
                if cap:
                    lines = lines.iloc[: int(cap)]
                elif osm_cfg.get("roads_feature_cap") is not None:
                    lines = lines.iloc[: int(osm_cfg["roads_feature_cap"])]
                hw = "highway" if "highway" in lines.columns else None
                rl, rw = accumulate_road_length_grid(
                    lines, hw, ref, osm_cfg, sample_m=float(osm_cfg.get("sample_m", 120.0))
                )
                out["_road_length_raw"] = rl
                out["_weighted_road_raw"] = rw
            except Exception as exc:
                warnings.warn(f"OSM roads read failed: {exc}", stacklevel=1)
                out["_road_length_raw"] = zeros.copy()
                out["_weighted_road_raw"] = zeros.copy()
        else:
            out["_road_length_raw"] = zeros.copy()
            out["_weighted_road_raw"] = zeros.copy()

    out["transport_area"] = out["industry_osm"].copy()
    out["road_length"] = out["_road_length_raw"]
    out["weighted_road_length"] = out["_weighted_road_raw"]
    return out
