"""Load fugitive_area.yaml and attach project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_fugitive_yaml(path: Path | None = None) -> dict[str, Any]:
    root = project_root()
    p = path or (Path(__file__).resolve().parent / "config" / "fugitive_area.yaml")
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Fugitive config not found: {p}")
    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid fugitive YAML")
    cfg["_project_root"] = root
    cfg["_config_path"] = p
    return cfg


def resolve(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)
