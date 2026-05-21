from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd

from . import common
from . import log


@dataclass
class RunContext:
    """Shared state for one create_osm_sector_packages run (bbox PBF reuse)."""

    root: Path
    pbf: Path
    nuts: Path
    cntr_code: str | None
    osmium_exe: str | None
    no_bbox_extract: bool
    allow_large_pbf: bool
    boundary: gpd.GeoDataFrame | None = None
    boundary_wgs: gpd.GeoDataFrame | None = None
    bbox_wgs84: str | None = None
    bbox_pbf: Path | None = None
    temp_root: Path | None = None
    filter_cache: dict[str, Path] = field(default_factory=dict)


def _filter_key(filters: tuple[str, ...]) -> str:
    """Hash osmium tag-filter tuple for cache lookup."""
    return hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:16]


def ensure_boundary(ctx: RunContext) -> None:
    """Load and cache NUTS boundary for the run country code."""
    if ctx.boundary is not None:
        return
    ctx.boundary, _n = common.load_boundary(ctx.nuts, ctx.cntr_code)
    ctx.bbox_wgs84 = common.bbox_str_wgs84(ctx.boundary)
    ctx.boundary_wgs = ctx.boundary.to_crs(4326)


def ensure_bbox_pbf(ctx: RunContext) -> Path:
    """Extract or reuse country bbox PBF; return path to use for parsing."""
    ensure_boundary(ctx)
    if ctx.bbox_pbf is not None and ctx.bbox_pbf.is_file():
        return ctx.bbox_pbf
    if ctx.no_bbox_extract or not ctx.osmium_exe:
        if (
            not ctx.osmium_exe
            and ctx.pbf.stat().st_size > common.MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL
            and not ctx.allow_large_pbf
        ):
            raise SystemExit(
                f"PBF is large and osmium-tool was not found. Install osmium-tool or set "
                f"ALLOW_LARGE_PBF_WITHOUT_OSMIUM=True."
            )
        if not ctx.osmium_exe:
            warnings.warn("osmium-tool not found: full PBF read is slow / high RAM.", stacklevel=1)
        return ctx.pbf

    t0 = log.Timer()
    if ctx.temp_root is None:
        ctx.temp_root = Path(tempfile.mkdtemp(prefix="osm_run_"))
    out = ctx.temp_root / "shared_bbox_extract.osm.pbf"
    log.info(f"bbox extract {ctx.bbox_wgs84} ...")
    common.extract_bbox(ctx.osmium_exe, ctx.bbox_wgs84, ctx.pbf, out)
    ctx.bbox_pbf = out
    log.info(f"bbox extract -> {log.format_mib(out)} ({log.format_duration(t0.elapsed())})")
    return out


def prepare_work_pbf(
    ctx: RunContext,
    *,
    prefilter_tags: bool,
    osmium_tag_filters: list[str] | tuple[str, ...] | None,
    bbox_extract_pbf: Path | None = None,
) -> Path:
    """Return PBF path after optional shared bbox extract and tags-filter."""
    if bbox_extract_pbf is not None:
        work = bbox_extract_pbf.expanduser().resolve()
        if not work.is_file():
            raise SystemExit(f"bbox_extract_pbf not found: {work}")
        return work

    work = ensure_bbox_pbf(ctx)
    if not prefilter_tags or not osmium_tag_filters or not ctx.osmium_exe:
        return work

    filt = tuple(str(x) for x in osmium_tag_filters)
    key = _filter_key(filt)
    if key in ctx.filter_cache and ctx.filter_cache[key].is_file():
        return ctx.filter_cache[key]

    if ctx.temp_root is None:
        ctx.temp_root = Path(tempfile.mkdtemp(prefix="osm_run_"))
    out = ctx.temp_root / f"tags_filtered_{key}.osm.pbf"
    t0 = log.Timer()
    log.info(f"tags-filter ({len(filt)} expressions) ...")
    common.tags_filter(ctx.osmium_exe, work, out, filt)
    ctx.filter_cache[key] = out
    log.info(f"tags-filter -> {log.format_mib(out)} ({log.format_duration(t0.elapsed())})")
    return out


def cleanup_run_context(ctx: RunContext, *, keep_temp: bool) -> None:
    """Remove temporary directory unless keep_temp is set."""
    if ctx.temp_root is not None and not keep_temp:
        shutil.rmtree(ctx.temp_root, ignore_errors=True)
    elif ctx.temp_root is not None and keep_temp:
        log.info(f"kept temp dir: {ctx.temp_root}")


def postprocess_layers(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    layer_order: list[str],
    boundary_wgs: gpd.GeoDataFrame,
    min_m2: float,
    layer_buffers_m: dict[str, float] | None = None,
) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Clip, dedupe, filter area, and buffer bucket rows into GeoDataFrames per layer."""
    layer_buffers_m = layer_buffers_m or {}
    out_layers: list[tuple[str, gpd.GeoDataFrame]] = []
    for name in layer_order:
        rows = buckets.get(name) or []
        if not rows:
            out_layers.append((name, common.empty_gdf_wgs84()))
            continue
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        gdf = common.dedupe_osm_id(gdf)
        gt = set(gdf.geometry.geom_type.unique())
        if gt <= {"Point"}:
            gdf = common.clip_points_to_3035(gdf, boundary_wgs)
        elif gt <= {"LineString", "MultiLineString"}:
            gdf = common.clip_lines_to_3035(gdf, boundary_wgs)
        else:
            gdf = common.clip_mixed_to_3035(gdf, boundary_wgs)
        gdf = common.filter_polygon_min_area(gdf, min_m2)
        buf_m = float(layer_buffers_m.get(name, 0) or 0)
        if buf_m > 0 and not gdf.empty:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(buf_m)
            gdf = common.filter_polygon_min_area(gdf, min_m2)
        out_layers.append((name, gdf))
    return out_layers


def write_sector_gpkg(
    out: Path,
    layers: list[tuple[str, gpd.GeoDataFrame]],
    *,
    sector_id: str,
    allow_all_empty: bool = False,
) -> dict[str, int]:
    """Write layers to GeoPackage and return feature counts per layer."""
    counts = {n: len(g) for n, g in layers}
    nonempty = [(n, g) for n, g in layers if not g.empty]
    if not nonempty and not allow_all_empty:
        raise SystemExit(
            f"[{sector_id}] all layers empty after processing; check PBF, NUTS mask, and rules."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    to_write = nonempty if nonempty else layers[:1]
    name0, gdf0 = to_write[0]
    gdf0.to_file(out, layer=name0, driver="GPKG", mode="w")
    for name, gdf in to_write[1:]:
        gdf.to_file(out, layer=name, driver="GPKG", mode="a")
    log.sector_info(sector_id, f"wrote {out.name} layers={counts}")
    return counts
