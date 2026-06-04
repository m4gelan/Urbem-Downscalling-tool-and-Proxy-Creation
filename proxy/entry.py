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

#SECTORS_ENABLED = ["E_Solvents"]
SECTORS_ENABLED = ["A_PublicPower", "B_Industry", "J_Waste"]

# Select area and or point matching
AREA_WEIGHTS = True
POINT_MATCHING = False

# Select run mode
# Options: build | prong_a | build_and_export | export_and_prong_a
# build: only build the weights
# prong_a: only run the prong_a analysis
# build_and_export: build the weights and export the W_groups
# export_and_prong_a: export the W_groups and run the prong_a analysis
RUN_MODE = "build_and_export" 

EXPORT_W_GROUPS = False
W_GROUPS_EXPORT_ROOT = "OUTPUT/Proxy_diagnostics/W_groups"
PRONG_A_SECTORS = "multi_group"
PRONG_A_W_SECTORS = "mix_export"

# Select country to build the weitghs and city for debug maps
COUNTRY = "Greece"
CITY = "Athens" # City for debug maps

# Info skips all, debug creates maps and other logs
LOG_LEVEL = "INFO" # INFO | DEBUG option
MAP_TYPE = 'FIXED_IMAGE' # INTERACTIVE for html map, FIXED_IMAGE for png maps for debug maps

EPSG_CRS = "EPSG:3035"
RESOLUTION_M = 100.0
PAD_M = 10.0

# Paths to the config files and the export root for the W_groups
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

def _run_build_loop(root: Path, table: dict, *, export_w_groups: bool) -> int:
    allowed = set(SECTORS)
    export_root = root / W_GROUPS_EXPORT_ROOT.replace("\\", "/")
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
                area_weights=AREA_WEIGHTS,
                point_matching=POINT_MATCHING,
                country_profile=resolve_country_profile(COUNTRY),
                crs=EPSG_CRS,
                resolution_m=RESOLUTION_M,
                pad_m=PAD_M,
                area_weights_viz_bbox_wgs84=BOUNDING_BOX if sk == "F_Roads" else viz_bbox,
                export_w_groups=export_w_groups,
                w_groups_export_root=export_root if export_w_groups else None,
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

def _run_prong_a(root: Path) -> int:
    from proxy.diagnostics.weight_sensitivity.run import load_prong_a_settings, run_prong_a
    cfg = load_prong_a_settings(root)
    year = int(cfg["year"])
    run_prong_a(
        root,
        country=COUNTRY,
        year=year,
        sector_keys=PRONG_A_SECTORS,
        active_eps=float(cfg["active_eps"]),
        similarity_threshold=float(cfg["similarity_threshold"]),
    )
    return 0


def _run_prong_a_w(root: Path) -> int:
    from proxy.diagnostics.weight_sensitivity.run import load_prong_a_w_settings, run_prong_a_w
    cfg = load_prong_a_w_settings(root)
    year = int(cfg["year"])
    run_prong_a_w(
        root,
        country=COUNTRY,
        year=year,
        sector_keys=cfg.get("sector_keys", PRONG_A_W_SECTORS),
        active_eps=float(cfg["active_eps"]),
        similarity_threshold=float(cfg["similarity_threshold"]),
    )
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
    mode = str(RUN_MODE).strip().lower()
    if mode == "build":
        return _run_build_loop(root, table, export_w_groups=EXPORT_W_GROUPS)
    if mode == "prong_a":
        return _run_prong_a(root)
    if mode == "prong_a_w":
        return _run_prong_a_w(root)
    if mode == "build_and_export":
        return _run_build_loop(root, table, export_w_groups=True)
    if mode == "export_and_prong_a":
        rc = _run_build_loop(root, table, export_w_groups=True)
        if rc != 0:
            return rc
        return _run_prong_a(root)
    if mode == "export_and_prong_a_w":
        rc = _run_build_loop(root, table, export_w_groups=True)
        if rc != 0:
            return rc
        return _run_prong_a_w(root)
    log.error(f"unknown RUN_MODE {RUN_MODE!r}")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
