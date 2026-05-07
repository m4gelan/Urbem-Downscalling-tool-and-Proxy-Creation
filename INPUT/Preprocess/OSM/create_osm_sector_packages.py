#!/usr/bin/env python3
"""
Build OSM sector GeoPackages under ``INPUT/Proxy/OSM/`` from ``osm_sector_layers.yaml``.

* ``mode: rules`` sectors use ``osm_engine`` (see ``osm_engine/osm_schema.yaml``), including ``aviation``.
* ``mode: plugin`` sectors use ``osm_engine.plugins.industry_v1`` / ``fugitive_v1``.

Usage (from repo root)::

    python INPUT/Preprocess/OSM/create_osm_sector_packages.py --country EL
    python INPUT/Preprocess/OSM/create_osm_sector_packages.py --country EL --sector waste
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def osm_preprocess_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_under_root(root: Path, p: str | Path) -> Path:
    q = Path(p)
    return q.resolve() if q.is_absolute() else (root / q).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML: {path}")
    return data


def output_gpkg(root: Path, defaults: dict[str, Any], sector_id: str) -> Path:
    out_dir = resolve_under_root(root, defaults.get("output_dir", "INPUT/Proxy/OSM"))
    tmpl = str(defaults.get("output_name_template", "{id}_layers.gpkg"))
    return (out_dir / tmpl.format(id=sector_id)).resolve()


def sector_ids_from_run_config(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("sectors")
    if not isinstance(raw, list):
        raise SystemExit("osm_sector_layers.yaml: need list 'sectors'.")
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str):
            ids.append(item.strip())
        elif isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]).strip())
        else:
            raise SystemExit(f"Invalid sector entry: {item!r}")
    return ids


def sector_entry(cfg: dict[str, Any], sector_id: str) -> dict[str, Any]:
    for item in cfg.get("sectors") or []:
        if isinstance(item, dict) and str(item.get("id", "")).strip() == sector_id:
            return dict(item)
        if isinstance(item, str) and item.strip() == sector_id:
            return {"id": sector_id}
    return {"id": sector_id}


def schema_sector_mode(schema_path: Path, sector_id: str) -> tuple[str, str | None]:
    sch = load_yaml(schema_path)
    sec = (sch.get("sectors") or {}).get(sector_id)
    if not isinstance(sec, dict):
        raise SystemExit(f"osm_schema.yaml: missing sector {sector_id!r}")
    mode = str(sec.get("mode", "rules"))
    plugin = sec.get("plugin")
    return mode, str(plugin) if plugin else None


def build_plugin_argv(
    *,
    sector_id: str,
    root: Path,
    defaults: dict[str, Any],
    entry: dict[str, Any],
    country: str | None,
    osmium: str | None,
    no_bbox_extract: bool,
    allow_large_pbf: bool,
) -> list[str]:
    pbf = resolve_under_root(root, defaults["pbf"])
    nuts = resolve_under_root(root, defaults["nuts_gpkg"])
    out = output_gpkg(root, defaults, sector_id)
    argv: list[str] = [f"{sector_id}_plugin", "--pbf", str(pbf), "--nuts", str(nuts), "--out", str(out)]
    if country:
        argv.extend(["--cntr-code", country.strip().upper()])
    if osmium:
        argv.extend(["--osmium", osmium])
    if no_bbox_extract:
        argv.append("--no-bbox-extract")
    if allow_large_pbf:
        argv.append("--allow-large-pbf-without-osmium")

    if sector_id == "industry":
        if entry.get("prefilter_tags", defaults.get("industry_prefilter_tags", False)) is True:
            argv.append("--prefilter-tags")
        if entry.get("keep_temp") is True:
            argv.append("--keep-temp")
        if entry.get("no_progress") is True:
            argv.append("--no-progress")
    elif sector_id == "fugitive":
        if entry.get("prefilter_tags", defaults.get("fugitive_prefilter_tags", False)) is True:
            argv.append("--prefilter-tags")
        if entry.get("keep_temp") is True:
            argv.append("--keep-temp")
        layer = entry.get("gpkg_layer")
        if layer:
            argv.extend(["--layer", str(layer)])
        argv.extend(["--min-polygon-area-m2", str(float(entry.get("min_polygon_area_m2", 10.0)))])
        if entry.get("dedupe_geometry") is True:
            argv.append("--dedupe-geometry")
        if entry.get("debug") is True:
            argv.append("--debug")
    else:
        raise SystemExit(f"Unknown plugin sector: {sector_id!r}")

    return argv


def run_plugin_main(plugin_name: str, argv: list[str]) -> None:
    mod = importlib.import_module(f"osm_engine.plugins.{plugin_name}")
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old


def main() -> int:
    here = osm_preprocess_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    root = project_root()

    default_cfg = here / "osm_sector_layers.yaml"
    schema_path = here / "osm_engine" / "osm_schema.yaml"

    p = argparse.ArgumentParser(description="Build OSM sector GeoPackages (unified engine).")
    p.add_argument("--config", type=Path, default=default_cfg)
    p.add_argument("--country", type=str, default=None)
    p.add_argument("--sector", type=str, default="all")
    p.add_argument("--osmium", type=str, default=None)
    p.add_argument("--no-bbox-extract", action="store_true")
    p.add_argument("--allow-large-pbf-without-osmium", action="store_true")
    args = p.parse_args()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    if not schema_path.is_file():
        raise SystemExit(f"Schema not found: {schema_path}")

    run_cfg = load_yaml(cfg_path)
    defaults = run_cfg.get("defaults") or {}
    if not defaults:
        raise SystemExit("osm_sector_layers.yaml must define defaults.")

    configured = sector_ids_from_run_config(run_cfg)
    want = args.sector.strip().lower()
    todo = configured if want == "all" else [want]
    for sid in todo:
        if sid not in configured:
            raise SystemExit(f"Sector {sid!r} not in {cfg_path}")

    for sid in todo:
        entry = sector_entry(run_cfg, sid)
        mode, plugin = schema_sector_mode(schema_path, sid)
        print(f"[{sid}] mode={mode}", flush=True)
        if mode == "plugin":
            if not plugin:
                raise SystemExit(f"Sector {sid} has mode plugin but no plugin: key")
            run_plugin_main(
                plugin,
                build_plugin_argv(
                    sector_id=sid,
                    root=root,
                    defaults=defaults,
                    entry=entry,
                    country=args.country,
                    osmium=args.osmium,
                    no_bbox_extract=bool(args.no_bbox_extract),
                    allow_large_pbf=bool(args.allow_large_pbf_without_osmium),
                ),
            )
        else:
            from osm_engine.run import run_rules_sector

            run_rules_sector(
                sid,
                pbf=resolve_under_root(root, defaults["pbf"]),
                nuts=resolve_under_root(root, defaults["nuts_gpkg"]),
                out=output_gpkg(root, defaults, sid),
                cntr_code=args.country,
                osmium_exe=args.osmium,
                no_bbox_extract=bool(args.no_bbox_extract),
                allow_large_pbf=bool(args.allow_large_pbf_without_osmium),
                bbox_extract_pbf=resolve_under_root(root, entry["bbox_extract_pbf"])
                if entry.get("bbox_extract_pbf")
                else None,
                with_optional=bool(entry.get("with_optional", False)),
                include_wastewater_plant=entry.get("include_wastewater_plant", True) is not False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
