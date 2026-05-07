"""Load YAML config and resolve paths relative to project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_TOP_KEYS = ("paths", "output", "cams", "ceip")


def project_root() -> Path:
    """PDM_local repo root: .../Waste/j_waste_weights/config_loader.py -> parents[2]."""
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load ``config.yaml`` from this package if ``path`` is None.
    Validates presence of top-level sections; does not require files to exist on disk.
    """
    root = project_root()
    cfg_path = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    for k in _REQUIRED_TOP_KEYS:
        if k not in cfg:
            raise KeyError(f"Config missing required top-level key {k!r} in {cfg_path}")
    cfg["_project_root"] = root
    cfg["_config_path"] = cfg_path
    return cfg


def resolve_cfg_path(cfg: dict[str, Any], key: str, *subkeys: str) -> Path:
    """Navigate cfg[subkeys] and resolve path against project root."""
    root: Path = cfg["_project_root"]
    cur: Any = cfg
    for sk in subkeys:
        if sk not in cur:
            raise KeyError(".".join(subkeys))
        cur = cur[sk]
    if not isinstance(cur, (str, Path)):
        raise TypeError(f"Expected path string at {'/'.join(subkeys)}, got {type(cur)}")
    return resolve_path(root, cur)


