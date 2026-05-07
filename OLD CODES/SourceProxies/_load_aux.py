"""Load sibling PublicPower auxiliary modules by path (not installed as packages)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_module_at(path: Path, logical_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(logical_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def load_cams_area_downscale(root: Path) -> ModuleType:
    p = root / "PublicPower" / "auxiliaries" / "cams_area_downscale_corine_landscan.py"
    return load_module_at(p, "cams_area_downscale_corine_landscan")


def load_cams_a_publicpower(root: Path) -> ModuleType:
    p = root / "PublicPower" / "auxiliaries" / "cams_A_publicpower_greece.py"
    return load_module_at(p, "cams_A_publicpower_greece")
