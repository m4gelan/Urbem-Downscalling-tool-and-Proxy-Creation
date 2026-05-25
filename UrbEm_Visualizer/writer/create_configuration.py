from __future__ import annotations

from pathlib import Path

import yaml

from UrbEm_Visualizer.dataset_loaders.check import check_input, load_expected
from UrbEm_Visualizer.paths import project_root


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return data


def _optional_sector_ids() -> set[str]:
    spec = load_expected()
    return set((spec.get("optional_sectors") or {}).keys())


def new_writer_config(country: str) -> dict:
    spec = load_expected()
    return {
        "country": country,
        "year": int(spec["year"]),
        "input_root": str(spec["input_root"]),
        "paths": {
            "cams": spec["cams"]["path"],
            "proxy_weights_root": spec["proxy_weights_root"],
        },
        "sectors": {},
        "absent_sources": [],
        "pollutants": [],
        "domain": None,
        "output": None,
    }


def save_run_config(config: dict, name: str) -> Path:
    from UrbEm_Visualizer.paths import runs_dir

    stem = str(name).strip()
    if not stem:
        raise ValueError("configuration name is required")
    if any(c in r'\/:*?"<>|' for c in stem):
        raise ValueError("configuration name contains invalid characters")
    runs_dir().mkdir(parents=True, exist_ok=True)
    path = runs_dir() / f"{stem}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path.resolve()


def config_from_check(
    country: str,
    root: Path | None = None,
    absent_sources: list[dict] | None = None,
) -> dict:
    result = check_input(country, root=root, absent_sources=absent_sources)
    if not result["ok"]:
        raise ValueError(result["message"])

    root = root or project_root()
    cfg = new_writer_config(country)
    cfg["absent_sources"] = list(absent_sources or [])
    optional_ids = _optional_sector_ids()

    for item in result["ok_items"]:
        if item["kind"] == "cams":
            cfg["paths"]["cams"] = _rel(root, item["path"])
            continue
        sector = item["sector"]
        if sector is None:
            continue
        role = item["kind"]
        cfg["sectors"].setdefault(sector, {})[role] = {"path": _rel(root, item["path"])}

    for item in result.get("accepted_absent") or []:
        sector = item["sector"]
        role = item["role"]
        if sector in optional_ids:
            continue
        cfg["sectors"].setdefault(sector, {})[role] = {"absent": True}

    return cfg


def apply_manual_paths(config: dict, manual: dict) -> dict:
    out = dict(config)
    paths = dict(out.get("paths") or {})
    if manual.get("cams"):
        paths["cams"] = manual["cams"]
    out["paths"] = paths

    optional_ids = _optional_sector_ids()
    absent: list[dict] = list(out.get("absent_sources") or [])
    absent_keys = {(a["sector"], a["role"]) for a in absent if a.get("sector") and a.get("role")}

    sectors = dict(out.get("sectors") or {})
    for sector_id, roles in (manual.get("sectors") or {}).items():
        if sector_id in optional_ids and roles.get("area_weights") == "__absent__":
            sectors.pop(sector_id, None)
            key = (sector_id, "area_weights")
            absent = [a for a in absent if (a["sector"], a["role"]) != key]
            absent_keys.discard(key)
            if roles.get("point_source") == "__absent__":
                key2 = (sector_id, "point_source")
                absent = [a for a in absent if (a["sector"], a["role"]) != key2]
                absent_keys.discard(key2)
            continue

        sec = dict(sectors.get(sector_id) or {})
        for role in ("point_source", "area_weights"):
            if role not in roles:
                continue
            val = roles[role]
            if val == "__absent__":
                if sector_id in optional_ids:
                    sec.pop(role, None)
                    key = (sector_id, role)
                    absent = [a for a in absent if (a["sector"], a["role"]) != key]
                    absent_keys.discard(key)
                else:
                    sec[role] = {"absent": True}
                    key = (sector_id, role)
                    if key not in absent_keys:
                        absent.append({"sector": sector_id, "role": role})
                        absent_keys.add(key)
            else:
                sec[role] = {"path": val}
                key = (sector_id, role)
                if key in absent_keys:
                    absent = [a for a in absent if (a["sector"], a["role"]) != key]
                    absent_keys.discard(key)
        if sec:
            sectors[sector_id] = sec
        else:
            sectors.pop(sector_id, None)
    out["sectors"] = sectors
    out["absent_sources"] = absent
    return out


def _rel(root: Path, path_str: str) -> str:
    p = Path(path_str)
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")
