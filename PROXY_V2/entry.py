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

from PROXY_V2.core import log
from PROXY_V2.core.alias import resolve_country_profile

# Below is the list of sectors that were created in the PROXY_V2 project
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
# To Enable Proxy creation for a specific sector, add the sector key to the list below
SECTORS_ENABLED = ["K_Agriculture"]
# Select True or false to enable or disable area weights
AREA_WEIGHTS = True
# Select True or false to enable or disable point matching (not all sectors support point matching)
POINT_MATCHING = True
# Select the country for the proxy creation
COUNTRY = "Greece"
CITY = "Thessaloniki"
# Select the log level for the proxy creation, 
# When DEBUG is selected, maps will be created
# 1 for point matching, 1 for area weights with the bounding box of the city selected

LOG_LEVEL = "DEBUG"  # DEBUG | INFO | WARNING | ERROR

EPSG_CRS = "EPSG:3035"
RESOLUTION_M = 100.0 # 100 m resolution for out output grids
PAD_M = 10.0 # 10 m padding around the points

if LOG_LEVEL == "DEBUG":
    bouding_box_yaml = yaml.safe_load(open("PROXY_V2/visualizers/bouding_boxes.yaml", "r"))
    bbox = bouding_box_yaml["bounding_boxes"].get(CITY)
    if not bbox:
        raise ValueError(f"City '{CITY}' not found in bounding_boxes.yaml")
    BOUNDING_BOX = tuple(bbox)
    filepaths_path = "PROXY_V2/config/filepaths.yaml"
else:
    BOUNDING_BOX = None

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
    """Drop a finished sector module and run GC so the next sector starts with a clean heap."""
    if module_name:
        sys.modules.pop(module_name, None)
    gc.collect()


def main() -> int:
    log.configure(LOG_LEVEL)
    viz_bbox = BOUNDING_BOX if log.debug_enabled() else None
    root = Path(__file__).resolve().parents[1]

    with open(root / filepaths_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    table = doc.get("Sector_specific") or doc.get("Sector_specific ")
    if not isinstance(table, dict):
        log.error("filepaths.yaml: missing Sector_specific")
        return 1

    allowed = set(SECTORS)
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
                area_weights_viz_bbox_wgs84=viz_bbox,
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

    _release_python_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
