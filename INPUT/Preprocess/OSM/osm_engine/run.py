from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import classify
from . import log
from . import pipeline
from . import pyosmium_io
from .rules_handler import RulesCollector


def _schema_path() -> Path:
    """Return path to osm_schema.yaml."""
    return Path(__file__).resolve().parent / "osm_schema.yaml"


def load_schema() -> dict[str, Any]:
    """Load osm_schema.yaml and merge industry classify_rules from sidecar file."""
    p = _schema_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    ind_path = Path(__file__).resolve().parent / "industry_classify_rules.yaml"
    if ind_path.is_file():
        rules = yaml.safe_load(ind_path.read_text(encoding="utf-8"))
        sec = (data.get("sectors") or {}).get("industry")
        if isinstance(sec, dict) and isinstance(rules, list):
            sec["classify_rules"] = rules
    return data


def _sector_ctx(sector_entry: dict[str, Any]) -> dict[str, Any]:
    """Build rules-engine context flags from per-sector run config entry."""
    return {
        "with_optional": bool(sector_entry.get("with_optional", False)),
        "include_wastewater_plant": sector_entry.get("include_wastewater_plant", True) is not False,
    }


def _offroad_sets(sec: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Parse offroad_sets from sector schema into frozensets."""
    raw = sec.get("offroad_sets") or {}
    return {k: frozenset(str(x) for x in v) for k, v in raw.items()}


def _store_osm_tags(sec: dict[str, Any], sector_entry: dict[str, Any]) -> bool:
    """Whether to embed full osm_tags JSON (off by default; huge RAM cost)."""
    if "store_osm_tags" in sector_entry:
        return sector_entry["store_osm_tags"] is True
    return sec.get("store_osm_tags") is True


def run_sector(
    sector_id: str,
    *,
    run_ctx: pipeline.RunContext,
    sector_entry: dict[str, Any],
    defaults: dict[str, Any],
    out: Path,
) -> None:
    """Run one sector (rules or classify) and write its GeoPackage."""
    data = load_schema()
    glob = data.get("global") or {}
    sectors = data.get("sectors") or {}
    if sector_id not in sectors:
        raise SystemExit(f"Unknown sector in osm_schema.yaml: {sector_id!r}")
    sec = dict(sectors[sector_id])
    sec["id"] = sector_id
    mode = str(sec.get("mode", "rules"))
    if mode not in ("rules", "classify"):
        raise SystemExit(f"Sector {sector_id!r}: mode must be rules or classify (got {mode!r})")

    min_m2 = float(sector_entry.get("min_polygon_area_m2", glob.get("min_polygon_area_m2", 10.0)))
    filters = sector_entry.get("osmium_tag_filters") or sec.get("osmium_tag_filters")
    if filters is not None:
        filters = list(filters)
    # Default: prefilter when schema defines osmium_tag_filters (unless explicitly disabled).
    prefilter = sector_entry.get("prefilter_tags")
    if prefilter is None:
        prefilter = bool(filters)
    else:
        prefilter = prefilter is True
    bbox_override = None
    if sector_entry.get("bbox_extract_pbf"):
        bbox_override = (run_ctx.root / str(sector_entry["bbox_extract_pbf"])).resolve()

    work_pbf = pipeline.prepare_work_pbf(
        run_ctx,
        prefilter_tags=prefilter,
        osmium_tag_filters=filters,
        bbox_extract_pbf=bbox_override,
    )
    pipeline.ensure_boundary(run_ctx)
    assert run_ctx.boundary_wgs is not None

    sec["store_osm_tags"] = _store_osm_tags(sec, sector_entry)
    pyosmium_idx = pyosmium_io.pick_pyosmium_idx(work_pbf, defaults, sector_entry)

    if mode == "classify":
        classify.run_classify_sector(
            sector_id,
            sec,
            work_pbf=work_pbf,
            boundary_wgs=run_ctx.boundary_wgs,
            out=out,
            min_m2=min_m2,
            pyosmium_idx=pyosmium_idx,
            osmium_exe=run_ctx.osmium_exe,
            defaults=defaults,
            sector_entry=sector_entry,
        )
        return

    ctx = _sector_ctx(sector_entry)
    off_sets = _offroad_sets(sec) if sec.get("offroad_sets") else {}
    t0 = log.Timer()
    log.sector_info(sector_id, f"parse start {work_pbf.name} idx={pyosmium_idx}")
    col = RulesCollector(sec, ctx=ctx, offroad_sets=off_sets)
    used_idx = pyosmium_io.apply_file(
        col,
        work_pbf,
        sector_id=sector_id,
        idx=pyosmium_idx,
        osmium_exe=run_ctx.osmium_exe,
        defaults=defaults,
        sector_entry=sector_entry,
    )
    if used_idx != pyosmium_idx:
        log.sector_info(sector_id, f"parse using idx={used_idx}")
    log.sector_info(
        sector_id,
        f"parse done kept={col.kept_count()} ({log.format_duration(t0.elapsed())})",
    )
    if log.debug_enabled():
        log.sector_debug(sector_id, f"per-layer counts={col.layer_counts()}")

    layer_order = list(sec.get("layer_order") or sorted(col.buckets.keys()))
    layer_buffers_m = sec.get("layer_buffers_m") or {}
    layers = pipeline.postprocess_layers(
        dict(col.buckets),
        layer_order=layer_order,
        boundary_wgs=run_ctx.boundary_wgs,
        min_m2=min_m2,
        layer_buffers_m={str(k): float(v) for k, v in layer_buffers_m.items()},
    )
    allow_empty = bool(sec.get("allow_all_empty_layers"))
    pipeline.write_sector_gpkg(out, layers, sector_id=sector_id, allow_all_empty=allow_empty)
