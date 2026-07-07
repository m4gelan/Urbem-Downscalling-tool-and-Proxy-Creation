from __future__ import annotations

from pathlib import Path

import yaml

from UrbEm_Visualizer.dataset_loaders.check import load_expected
from UrbEm_Visualizer.paths import project_root

_SECTOR_FOLDER_ALIAS = {"C_OtherCombustion": "C_Othercombustion"}

_LABELS = {
    "A_PublicPower": "Public power",
    "B_Industry": "Industry",
    "C_OtherCombustion": "Other combustion",
    "D_Fugitive": "Fugitive",
    "E_Solvents": "Solvents",
    "G_Shipping": "Shipping",
    "H_Aviation": "Aviation",
    "I_Offroad": "Off-road",
    "J_Waste": "Waste",
    "K_Agriculture": "Agriculture",
    "F_Roads": "Roads",
}


def roads_category_names() -> list[str]:
    cfg = load_sector_yaml("F_Roads")
    return list((cfg.get("cams_f_categories") or {}).keys())


def sector_label(sector_id: str) -> str:
    return _LABELS.get(sector_id, sector_id.replace("_", " "))


def sector_folder(sector_id: str) -> str:
    return _SECTOR_FOLDER_ALIAS.get(sector_id, sector_id)


def sector_config_path(sector_id: str) -> Path:
    folder = sector_folder(sector_id)
    return project_root() / "proxy" / "config" / "sector" / folder / f"{folder}_sector_config.yaml"


def load_sector_yaml(sector_id: str) -> dict:
    path = sector_config_path(sector_id)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: invalid sector config")
    return data


def sector_order(config: dict) -> list[str]:
    spec = load_expected()
    order: list[str] = []
    sectors_cfg = config.get("sectors") or {}
    for sid in spec["sectors"]:
        if sid in sectors_cfg:
            order.append(sid)
    for sid in (spec.get("optional_sectors") or {}):
        if sid in sectors_cfg:
            order.append(sid)
    return order


def resolve_sector_id(name: str, config: dict) -> str:
    raw = str(name).strip()
    sectors_cfg = config.get("sectors") or {}
    if raw in sectors_cfg:
        return raw
    by_lower = {k.lower(): k for k in sectors_cfg}
    hit = by_lower.get(raw.lower())
    if hit:
        return hit
    for sid in sectors_cfg:
        if sector_folder(sid).lower() == raw.lower():
            return sid
    known = ", ".join(sorted(sectors_cfg))
    raise ValueError(f"sector {name!r} not in config (available: {known})")


def sector_mode(sector_id: str) -> str:
    spec = load_expected()
    if sector_id in spec["sectors"]:
        return spec["sectors"][sector_id]["mode"]
    return spec["optional_sectors"][sector_id]["mode"]
