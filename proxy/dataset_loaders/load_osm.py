from __future__ import annotations

# Read pre-built GPKG layers, restrict to CAMS country cells, buffer in metres,
# merge geometries, then rasterize onto the same grid as CORINE (e.g. for area weights).

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.crs import CRS as RioCRS
from rasterio.transform import array_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells

# load_osm runs three geometry passes on the same file; cache avoids reading the GPKG three times.
_LAYER_CACHE: dict[tuple[str, tuple[str, ...]], gpd.GeoDataFrame] = {}


def _read_osm_layers(osm_gpkg: Path, layer_order: list[str]) -> gpd.GeoDataFrame:
    """Stack named layers from one GeoPackage into a single GeoDataFrame (geometry + attributes)."""
    key = (str(osm_gpkg.resolve()), tuple(layer_order))
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]
    frames: list[gpd.GeoDataFrame] = []
    for name in layer_order:
        try:
            g = gpd.read_file(osm_gpkg, layer=name)
        except Exception as exc:
            log.warning("OSM: skip layer %r: %s", name, exc)
            continue
        if g is None or g.empty:
            continue
        if g.crs is None:
            raise ValueError(f"OSM layer {name!r} has no CRS ({osm_gpkg})")
        frames.append(g)
    if not frames:
        out = gpd.GeoDataFrame(geometry=[], crs=None)
    else:
        out = pd.concat(frames, ignore_index=True)
        if not isinstance(out, gpd.GeoDataFrame):
            out = gpd.GeoDataFrame(out, geometry="geometry", crs=frames[0].crs)
        elif out.crs is None:
            out = out.set_crs(frames[0].crs)
    _LAYER_CACHE[key] = out
    return out


def _cams_domain_metric(
    cams_cells: dict[int, dict[str, Any]],
    metric_crs: Any,
    domain_clip_buffer_m: float,
) -> Any:
    """Union of CAMS cell rectangles (WGS84 bounds) reprojected to metric CRS; optional outer buffer in metres."""
    polys = []
    for c in cams_cells.values():
        b = c["cell_bounds_wgs84"]
        polys.append(box(b["west"], b["south"], b["east"], b["north"]))
    dom = unary_union(polys)
    g = gpd.GeoDataFrame(geometry=[dom], crs="EPSG:4326").to_crs(metric_crs)
    geom = g.geometry.iloc[0]
    if domain_clip_buffer_m and domain_clip_buffer_m > 0:
        geom = geom.buffer(float(domain_clip_buffer_m))
    return geom


def _clip_geoms_to_domain_metric(
    raw: gpd.GeoDataFrame,
    domain_metric: Any,
    metric_crs: Any,
    geom_types: tuple[str, ...],
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Filter to *geom_types*, reproject to *metric_crs*, clip to *domain_metric*, optional positive buffer (m)."""
    if raw.empty:
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    mask_gdf = gpd.GeoDataFrame(geometry=[domain_metric], crs=metric_crs)
    # Multi-part rows become one row per part so Point vs MultiPoint filtering is consistent.
    g = raw.explode(ignore_index=True)
    m = g.geometry.geom_type.isin(geom_types)
    g = g.loc[m].copy()
    if g.empty:
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    g = g.to_crs(metric_crs)
    g = gpd.clip(g, mask_gdf)
    if g.empty:
        return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
    buf = float(buffer_m)
    if buf > 0:
        # Points/lines become polygons; polygons grow slightly so raster pixels near edges still catch them.
        g["geometry"] = g.geometry.buffer(buf)
    return g


def _clip_buffer_metric(
    raw: gpd.GeoDataFrame,
    cams_cells: dict[int, dict[str, Any]],
    osm_cfg: dict[str, Any],
    geom_types: tuple[str, ...],
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """Clip *raw* to the union of CAMS cells (WGS84 bounds → metric CRS) plus ``domain_clip_buffer_m``."""
    metric_crs = osm_cfg["metric_crs"]
    domain = _cams_domain_metric(cams_cells, metric_crs, float(osm_cfg["domain_clip_buffer_m"]))
    return _clip_geoms_to_domain_metric(raw, domain, metric_crs, geom_types, buffer_m)


def _osm_buffered_stack(
    raw: gpd.GeoDataFrame,
    cams_cells: dict[int, dict[str, Any]],
    osm_cfg: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Clip to CAMS domain, split by geometry type, apply per-type buffers (metres), concatenate."""
    bm = osm_cfg["buffer_m"]
    parts = [
        _clip_buffer_metric(raw, cams_cells, osm_cfg, ("Point", "MultiPoint"), float(bm["point"])),
        _clip_buffer_metric(raw, cams_cells, osm_cfg, ("LineString", "MultiLineString"), float(bm["line"])),
        _clip_buffer_metric(raw, cams_cells, osm_cfg, ("Polygon", "MultiPolygon"), float(bm["polygon"])),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=osm_cfg["metric_crs"])
    out = pd.concat(parts, ignore_index=True)
    if not isinstance(out, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=parts[0].crs)
    return out


def load_osm(
    osm_gpkg: Path,
    cams_cells: dict[int, dict[str, Any]] | None,
    osm_cfg: dict[str, Any],
    *,
    for_point_matching: bool = False,
    clip_domain_metric: Any | None = None,
) -> gpd.GeoDataFrame:
    """Load OSM features from *layer_order*, clipped in *metric_crs*.

    Default path: *cams_cells* is required; clip to CAMS country footprint, apply per-geometry buffers (rasterize).

    Point matching (*for_point_matching* true): only Polygon/MultiPolygon, no edge buffer. Pass
    *clip_domain_metric* (Shapely geometry in ``osm_cfg["metric_crs"]``, e.g. CAMS-point bbox + 15 km); *cams_cells* is ignored.
    """
    raw = _read_osm_layers(osm_gpkg, list(osm_cfg["layer_order"]))
    metric_crs = osm_cfg["metric_crs"]
    if for_point_matching:
        if clip_domain_metric is None:
            raise ValueError("load_osm(..., for_point_matching=True) requires clip_domain_metric")
        return _clip_geoms_to_domain_metric(
            raw, clip_domain_metric, metric_crs, ("Polygon", "MultiPolygon"), 0.0
        )

    return _osm_buffered_stack(raw, cams_cells, osm_cfg)


def _effective_osm_cfg_for_buffers(
    osm_cfg: dict[str, Any], buffer_m_override: dict[str, Any] | None
) -> dict[str, Any]:
    if not buffer_m_override:
        return osm_cfg
    cfg = dict(osm_cfg)
    base = dict(osm_cfg.get("buffer_m") or {})
    for k, v in buffer_m_override.items():
        base[str(k)] = float(v)
    cfg["buffer_m"] = base
    return cfg


def load_osm_filtered(
    osm_gpkg: Path,
    cams_cells: dict[int, dict[str, Any]],
    osm_cfg: dict[str, Any],
    *,
    column: str | None = None,
    values: frozenset[str] | None = None,
    match: dict[str, Any] | None = None,
    buffer_m_override: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Clip/buffer OSM after either *column*∈*values* (industry ids) or *match* ``any_of`` tag rules (fugitive)."""
    eff = _effective_osm_cfg_for_buffers(osm_cfg, buffer_m_override)
    raw = _read_osm_layers(osm_gpkg, list(eff["layer_order"]))
    if raw.empty:
        return gpd.GeoDataFrame(geometry=[], crs=eff["metric_crs"])

    has_ids = column is not None and values is not None
    has_match = match is not None and bool((match or {}).get("any_of"))
    if has_ids == has_match:
        raise ValueError(
            "load_osm_filtered: pass exactly one of (column, values) or match with non-empty any_of"
        )
    if has_match:
        rules = (match or {}).get("any_of") or []

        def row_ok(row: Any) -> bool:
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
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

        m = raw.apply(row_ok, axis=1)
        raw = raw.loc[m].copy()
    else:
        if column is None or values is None:
            raise ValueError("load_osm_filtered: column and values are required when match is not used")
        if column not in raw.columns:
            raise ValueError(
                f"OSM GeoPackage stack has no column {column!r} (have {sorted(raw.columns.tolist())})"
            )
        vs = {str(v) for v in values}
        raw = raw.loc[raw[column].astype(str).isin(vs)].copy()

    return _osm_buffered_stack(raw, cams_cells, eff)


def rasterize_osm(
    gdf: gpd.GeoDataFrame,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    rasterize_cfg: dict[str, Any],
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Grid aligned to CORINE: burn_value where geometry meets pixel (all_touched); 0 outside CAMS footprint."""
    fill = float(rasterize_cfg["fill"])
    burn = float(rasterize_cfg["burn_value"])
    dtype = np.dtype(rasterize_cfg["dtype"])
    all_touched = bool(rasterize_cfg["all_touched"])
    acc = np.full((height, width), fill, dtype=np.float32)
    if gdf is None or gdf.empty:
        return acc.astype(dtype, copy=False)

    dst_crs = RioCRS.from_user_input(raster_crs)
    if gdf.crs is not None and RioCRS.from_user_input(gdf.crs) == dst_crs:
        g = gdf
    else:
        g = gdf.to_crs(raster_crs)

    left, bottom, right, top = array_bounds(height, width, transform)
    g = gpd.clip(g, gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs=raster_crs))

    res = min(abs(float(transform.a)), abs(float(transform.e)))
    tol = res * 0.5 if res > 0 else 0.0

    def _burn_shapes():
        for geom in g.geometry:
            if geom is None or geom.is_empty:
                continue
            if tol > 0:
                geom = geom.simplify(tol, preserve_topology=True)
            yield (geom, burn)

    # Paint burn_value onto pixels touched by geometry (all_touched from config); other pixels stay fill.
    rio_features.rasterize(
        _burn_shapes(),
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        out=acc,
        dtype=np.float32,
        all_touched=all_touched,
        merge_alg=rasterio.enums.MergeAlg.replace,
    )
    # Match CORINE/EMODNET: zero pixels whose centre is not inside any CAMS cell for this country.
    inside = pixels_inside_cams_cells(height, width, transform, raster_crs, cams_cells)
    acc = np.where(inside, acc, fill)
    return acc.astype(dtype, copy=False)
