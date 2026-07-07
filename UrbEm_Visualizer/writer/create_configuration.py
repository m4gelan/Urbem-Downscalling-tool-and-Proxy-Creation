from __future__ import annotations

from pathlib import Path

import yaml

from UrbEm_Visualizer.dataset_loaders.check import (
    cams_path_for_emissions_year,
    check_input,
    default_emissions_year,
    infer_emissions_year_from_path,
    load_expected,
)
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


def new_writer_config(country: str, emissions_year: int | None = None) -> dict:
    spec = load_expected()
    emis_year = int(emissions_year) if emissions_year is not None else default_emissions_year(spec)
    return {
        "country": country,
        "year": emis_year,
        "emissions_year": emis_year,
        "input_root": str(spec["input_root"]),
        "paths": {
            "cams": cams_path_for_emissions_year(emis_year, spec),
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


def _apply_ok_items(cfg: dict, ok_items: list[dict], root: Path) -> None:
    for item in ok_items:
        if item["kind"] == "cams":
            cfg["paths"]["cams"] = _rel(root, item["path"])
            continue
        sector = item["sector"]
        if sector is None:
            continue
        role = item["kind"]
        if item.get("roads_categories"):
            cfg["sectors"].setdefault(sector, {})["area_weights"] = {
                "categories": {
                    cat: _rel(root, p) for cat, p in item["roads_categories"].items()
                },
            }
            continue
        cfg["sectors"].setdefault(sector, {})[role] = {"path": _rel(root, item["path"])}


def merge_check_paths(config: dict, check: dict | None = None, root: Path | None = None) -> dict:
    root = root or project_root()
    cfg = dict(config)
    cfg["sectors"] = dict(cfg.get("sectors") or {})
    if check is None:
        country = cfg.get("country")
        if not country:
            return cfg
        emis_year = cfg.get("emissions_year")
        if emis_year is None and (cfg.get("paths") or {}).get("cams"):
            emis_year = infer_emissions_year_from_path(cfg["paths"]["cams"])
        check = check_input(country, absent_sources=cfg.get("absent_sources"), emissions_year=emis_year)
    _apply_ok_items(cfg, check.get("ok_items") or [], root)
    if cfg.get("emissions_year") is None:
        cams_rel = (cfg.get("paths") or {}).get("cams")
        if cams_rel:
            inferred = infer_emissions_year_from_path(cams_rel)
            if inferred is not None:
                cfg["emissions_year"] = inferred
                cfg["year"] = inferred
    return cfg


def config_from_check(
    country: str,
    root: Path | None = None,
    absent_sources: list[dict] | None = None,
    emissions_year: int | None = None,
) -> dict:
    result = check_input(country, root=root, absent_sources=absent_sources, emissions_year=emissions_year)
    if not result["ok"]:
        raise ValueError(result["message"])

    root = root or project_root()
    cfg = new_writer_config(country, emissions_year=result["emissions_year"])
    cfg["absent_sources"] = list(absent_sources or [])
    optional_ids = _optional_sector_ids()
    _apply_ok_items(cfg, result["ok_items"], root)

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
        inferred = infer_emissions_year_from_path(manual["cams"])
        if inferred is not None:
            out["emissions_year"] = inferred
            out["year"] = inferred
        elif manual.get("emissions_year") is not None:
            emis_year = int(manual["emissions_year"])
            out["emissions_year"] = emis_year
            out["year"] = emis_year
    elif manual.get("emissions_year") is not None:
        emis_year = int(manual["emissions_year"])
        out["emissions_year"] = emis_year
        out["year"] = emis_year
        paths["cams"] = cams_path_for_emissions_year(emis_year)
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
