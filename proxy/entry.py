from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import yaml

from proxy.core import log
from proxy.core.alias import resolve_country_profile
from proxy.core.cams_year import cams_path_for_year, load_cams_inventory, load_patched_sector_config

SECTORS = [
    "A_PublicPower",
    "B_Industry",
    "C_Othercombustion",
    "D_Fugitive",
    "E_Solvents",
    "F_Roads",
    "G_Shipping",
    "H_Aviation",
    "I_Offroad",
    "J_Waste",
    "K_Agriculture",
]

SECTORS_ENABLED = ["K_Agriculture"]
# SECTORS_ENABLED = SECTORS

# Select area and or point matching
AREA_WEIGHTS = True
POINT_MATCHING = False

# Select countries to build — processed one by one
COUNTRIES = [
    "Italy",
]

CITY = "Cremona" # City for debug maps

# Info skips all, debug creates maps and other logs
LOG_LEVEL = "DEBUG" # INFO | DEBUG option
MAP_TYPE = 'INTERACTIVE' # INTERACTIVE for html map, FIXED_IMAGE for png maps for debug maps

EPSG_CRS = "EPSG:3035"
RESOLUTION_M = 100.0
PAD_M = 10.0

# CAMS inventory year for proxy weights and point matching (2019 | 2021)
CAMS_YEAR = 2021

filepaths_path = "proxy/config/filepaths.yaml"


_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

BOUNDING_BOX = None
if CITY:
    bouding_box_yaml = yaml.safe_load(open("proxy/visualizers/bouding_boxes.yaml", "r"))
    bbox = bouding_box_yaml["bounding_boxes"].get(CITY)
    if not bbox:
        raise ValueError(f"City '{CITY}' not found in bounding_boxes.yaml")
    BOUNDING_BOX = tuple(bbox)

def _format_duration(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    m, r = divmod(int(round(s)), 60)
    return f"{m}m{r:02d}s"

def _sector_output_tif(out_dir: Path) -> str:
    tifs = sorted(out_dir.glob("*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return tifs[0].name if tifs else "(no .tif yet)"

def _release_python_memory(module_name: str | None = None) -> None:
    if module_name:
        sys.modules.pop(module_name, None)
    gc.collect()

def _run_build_loop(root: Path, table: dict, *, country: str, cams: dict) -> int:
    allowed = set(SECTORS)
    total_time = 0
    for sector_key in SECTORS_ENABLED:
        _release_python_memory()
        sk = str(sector_key).strip()
        if sk not in allowed:
            log.error(f"This sector {sk!r} is not in the allowed list: {allowed}")
            continue
        sector_code = sk.split("_", 1)[0]
        mod_name = f"p_{sk}"
        log.info("--------------------------------")
        log.info(f"Processing sector: {sk}")
        log.info("--------------------------------")
        entry = table.get(sk)
        if not isinstance(entry, dict):
            log.error(f"skip {sk!r}: not in Sector_specific")
            continue
        config_rel = entry.get("config")
        out_rel = entry.get("output_folder")
        pipeline_rel = entry.get("pipeline")
        if not (isinstance(config_rel, str) and isinstance(out_rel, str) and isinstance(pipeline_rel, str)):
            log.error(f"skip {sk!r}: need config, output_folder, pipeline strings")
            continue
        cfg_path = root / config_rel.replace("\\", "/")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        sector_cfg = load_patched_sector_config(
            cfg_path,
            int(CAMS_YEAR),
            cams_path_for_year(cams, int(CAMS_YEAR)),
        )
        out_dir = root / out_rel.replace("\\", "/")
        out_dir.mkdir(parents=True, exist_ok=True)
        script = root / pipeline_rel.replace("\\", "/")
        viz_bbox = BOUNDING_BOX if log.debug_enabled() else None
        t0 = time.perf_counter()
        try:
            spec = importlib.util.spec_from_file_location(mod_name, script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.build(
                out_dir,
                cfg_path,
                sector_config=sector_cfg,
                area_weights=AREA_WEIGHTS,
                point_matching=POINT_MATCHING,
                country_profile=resolve_country_profile(country),
                crs=EPSG_CRS,
                resolution_m=RESOLUTION_M,
                pad_m=PAD_M,
                area_weights_viz_bbox_wgs84=BOUNDING_BOX if sk == "F_Roads" else viz_bbox,
            )
        except Exception:
            import traceback
            tb = traceback.format_exc(limit=3)
            log.error(f"sector {sk!r} failed:\n{tb}")
            return 1
        finally:
            _release_python_memory(mod_name)
        elapsed = time.perf_counter() - t0
        log.info(
            f"Sector : {sector_code}\n"
            f"file : {_sector_output_tif(out_dir)}\n"
            f"time : {_format_duration(elapsed)}"
        )
        total_time += elapsed
    if len(SECTORS_ENABLED) > 0:
        log.info("--------------------------------")
        log.info(f"All sectors processed successfully")
        log.info(f"Total sectors processed: {len(SECTORS_ENABLED)} | Total time: {_format_duration(total_time)}")
        log.info(f"--------------------------------")
    return 0

def main() -> int:
    log.configure(LOG_LEVEL)
    root = Path(__file__).resolve().parents[1]
    with open(root / filepaths_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    table = doc.get("Sector_specific") or doc.get("Sector_specific ")
    if not isinstance(table, dict):
        log.error("filepaths.yaml: missing Sector_specific")
        return 1
    if not COUNTRIES:
        log.error("COUNTRIES must be a non-empty list")
        return 1
    cams = load_cams_inventory(root / filepaths_path.replace("\\", "/"))
    log.info(f"CAMS inventory year: {int(CAMS_YEAR)}")
    for country in COUNTRIES:
        log.info("================================")
        log.info(f"Country: {country}")
        log.info("================================")
        rc = _run_build_loop(root, table, country=str(country).strip(), cams=cams)
        if rc != 0:
            log.error(f"Stopped after failure for country {country!r}")
            return rc
    if len(COUNTRIES) > 1:
        log.info(f"All {len(COUNTRIES)} countries processed successfully")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
