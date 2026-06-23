#!/usr/bin/env python3
"""Build OSM sector GeoPackages from osm_sector_layers.yaml + osm_engine/osm_schema.yaml."""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Top of file — user edits here only

SECTORS = ["waste", "solvents", "offroad", 'shipping', "industry", "fugitive", "aviation", "agricultural"]
SECTORS_ENABLED = ["offroad"]  #[s for s in SECTORS if s != "waste"]
COUNTRY = "Spain"  # maps to NUTS CNTR_CODE 
OUTPUT_DIR = f"INPUT/Proxy/OSM/{COUNTRY}"  # GeoPackage output folder (repo-relative or absolute)
LOG_LEVEL = "DEBUG"  # DEBUG | INFO | WARNING | ERROR
NO_BBOX_EXTRACT = False
MAINLAND_BBOX_EXTRACT = True  # FR/ES: NUTS2 mainland bbox when cutting from europe-latest
COUNTRY_PBF = None # e.g. INPUT/Preprocess/OSM/_source/france-latest.osm.pbf
TEMP_DIR = None  # e.g. G:/osm_temp — temp PBFs (needs ~15–25 GiB free for FR from europe)
ALLOW_LARGE_PBF_WITHOUT_OSMIUM = False
OSMIUM_EXE = None  # None → shutil.which("osmium")
PARSE_PROGRESS = True  # tqdm bar during pyosmium (pip install tqdm); else log every 500k objects

COUNTRY_TO_CNTR: dict[str, str] = {
    "Austria": "AT",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czechia": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "EL",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Netherlands": "NL",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
}


def project_root() -> Path:
    """Return repository root (parent of INPUT/)."""
    return Path(__file__).resolve().parents[3]


def osm_preprocess_dir() -> Path:
    """Return INPUT/Preprocess/OSM directory."""
    return Path(__file__).resolve().parent


def resolve_under_root(root: Path, p: str | Path) -> Path:
    """Resolve path relative to repo root unless already absolute."""
    q = Path(p)
    return q.resolve() if q.is_absolute() else (root / q).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file as a dict."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML: {path}")
    return data


def output_gpkg(root: Path, defaults: dict[str, Any], sector_id: str) -> Path:
    """Build output GeoPackage path for a sector id."""
    out_dir = resolve_under_root(root, defaults.get("output_dir", OUTPUT_DIR))
    tmpl = str(defaults.get("output_name_template", "{id}_layers.gpkg"))
    return (out_dir / tmpl.format(id=sector_id)).resolve()


def sector_entry(cfg: dict[str, Any], sector_id: str) -> dict[str, Any]:
    """Return per-sector run config entry, or minimal dict with id only."""
    for item in cfg.get("sectors") or []:
        if isinstance(item, dict) and str(item.get("id", "")).strip() == sector_id:
            return dict(item)
        if isinstance(item, str) and item.strip() == sector_id:
            return {"id": sector_id}
    return {"id": sector_id}


def cntr_code_for_country(country: str) -> str:
    """Map human country name to NUTS CNTR_CODE."""
    key = str(country).strip()
    if key.upper() in {v.upper() for v in COUNTRY_TO_CNTR.values()}:
        return key.upper()
    if key not in COUNTRY_TO_CNTR:
        raise SystemExit(f"Unknown COUNTRY {key!r}; add to COUNTRY_TO_CNTR in create_osm_sector_packages.py")
    return COUNTRY_TO_CNTR[key].upper()


def validate_startup(schema: dict[str, Any], run_cfg: dict[str, Any]) -> None:
    """Fail fast if enabled sectors or schema modes are invalid."""
    from osm_engine import log

    sectors_yaml = {
        str(item.get("id") if isinstance(item, dict) else item).strip()
        for item in (run_cfg.get("sectors") or [])
    }
    schema_sectors = schema.get("sectors") or {}
    allowed_modes = {"rules", "classify"}
    for sid in SECTORS_ENABLED:
        sk = str(sid).strip()
        if sk not in SECTORS:
            log.error(f"sector {sk!r} not in SECTORS list")
            raise SystemExit(1)
        if sk not in sectors_yaml:
            log.error(f"sector {sk!r} missing from osm_sector_layers.yaml")
            raise SystemExit(1)
        if sk not in schema_sectors:
            log.error(f"sector {sk!r} missing from osm_schema.yaml")
            raise SystemExit(1)
        mode = str((schema_sectors[sk] or {}).get("mode", "rules"))
        if mode not in allowed_modes:
            log.error(f"sector {sk!r}: invalid mode {mode!r}")
            raise SystemExit(1)
        if mode == "classify":
            rules = (schema_sectors[sk] or {}).get("classify_rules")
            ind_file = osm_preprocess_dir() / "osm_engine" / "industry_classify_rules.yaml"
            if not rules and sk == "industry" and not ind_file.is_file():
                log.error(f"sector {sk!r}: classify_rules missing (run build_industry_rules.py once)")
                raise SystemExit(1)


def main() -> int:
    """Run all enabled sectors and write GeoPackages."""
    here = osm_preprocess_dir()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from osm_engine import log
    from osm_engine import pipeline
    from osm_engine.run import load_schema, run_sector

    log.configure(LOG_LEVEL)
    root = project_root()
    run_cfg_path = here / "osm_sector_layers.yaml"
    run_cfg = load_yaml(run_cfg_path)
    schema = load_schema()
    validate_startup(schema, run_cfg)

    defaults = dict(run_cfg.get("defaults") or {})
    if not defaults:
        raise SystemExit("osm_sector_layers.yaml must define defaults.")
    defaults["output_dir"] = OUTPUT_DIR

    cntr = cntr_code_for_country(COUNTRY)
    pbf = resolve_under_root(root, COUNTRY_PBF or defaults["pbf"])
    nuts = resolve_under_root(root, defaults["nuts_gpkg"])
    temp_dir = resolve_under_root(root, TEMP_DIR) if TEMP_DIR else None

    from osm_engine import common

    osmium_exe = common.resolve_osmium_exe(OSMIUM_EXE)

    log.info(f"start country={COUNTRY} cntr={cntr} sectors={SECTORS_ENABLED}")
    t_run = time.perf_counter()

    run_ctx = pipeline.RunContext(
        root=root,
        pbf=pbf,
        nuts=nuts,
        cntr_code=cntr,
        osmium_exe=osmium_exe,
        no_bbox_extract=NO_BBOX_EXTRACT,
        allow_large_pbf=ALLOW_LARGE_PBF_WITHOUT_OSMIUM,
        mainland_bbox_extract=MAINLAND_BBOX_EXTRACT,
        temp_dir=temp_dir,
    )

    try:
        for sid in SECTORS_ENABLED:
            sk = str(sid).strip()
            entry = sector_entry(run_cfg, sk)
            if PARSE_PROGRESS:
                entry["show_parse_progress"] = True
            out = output_gpkg(root, defaults, sk)
            log.sector_info(sk, f"start -> {out.name}")
            t0 = time.perf_counter()
            try:
                run_sector(
                    sk,
                    run_ctx=run_ctx,
                    sector_entry=entry,
                    defaults=defaults,
                    out=out,
                )
            except SystemExit:
                raise
            except Exception as e:
                log.error(f"sector {sk!r} failed: {e}")
                raise
            log.sector_info(sk, f"done ({log.format_duration(time.perf_counter() - t0)})")
            gc.collect()

        log.info(f"all done {len(SECTORS_ENABLED)} sectors ({log.format_duration(time.perf_counter() - t_run)})")
        return 0
    finally:
        pipeline.cleanup_run_context(run_ctx, keep_temp=False)


if __name__ == "__main__":
    raise SystemExit(main())
