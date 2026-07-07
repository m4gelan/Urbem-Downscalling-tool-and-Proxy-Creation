from __future__ import annotations

import copy
from pathlib import Path

import yaml

_CAMS_BLOCKS = ("cams_point_sources", "cams_sector_cells", "cams_area_emissions")


def load_cams_inventory(filepaths_yaml: Path) -> dict:
    with open(filepaths_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    cams = doc.get("cams")
    if not isinstance(cams, dict) or "files" not in cams:
        raise KeyError(f"{filepaths_yaml}: missing cams.files")
    return cams


def cams_path_for_year(cams: dict, year: int) -> str:
    files = cams["files"]
    key = int(year)
    if key in files:
        return str(files[key])
    skey = str(key)
    if skey in files:
        return str(files[skey])
    raise KeyError(f"no CAMS file for year {key}")


def patch_sector_config(cfg: dict, year: int, cams_path: str) -> dict:
    out = copy.deepcopy(cfg)
    fp = out.setdefault("filepaths", {})
    cam = fp.get("CAMS")
    if not isinstance(cam, dict):
        cam = {}
    cam = dict(cam)
    cam["path"] = cams_path
    fp["CAMS"] = cam
    out["filepaths"] = fp
    y = int(year)
    for block in _CAMS_BLOCKS:
        b = out.get(block)
        if isinstance(b, dict) and "year" in b:
            b = dict(b)
            b["year"] = y
            out[block] = b
    return out


def load_patched_sector_config(src: Path, year: int, cams_path: str) -> dict:
    with src.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{src}: sector config must be a mapping")
    return patch_sector_config(cfg, year, cams_path)
