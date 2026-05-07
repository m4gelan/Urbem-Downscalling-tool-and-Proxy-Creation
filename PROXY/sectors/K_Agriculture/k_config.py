"""Project root and config helpers for K_Agriculture (replaces top-level ``Agriculture.config``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from PROXY.core.dataloaders.config import load_yaml as load_yaml_mapping
from PROXY.core.dataloaders.config import project_root as core_project_root

_REPO_ROOT = core_project_root(Path(__file__))
def project_root(cfg: dict[str, Any] | None) -> Path:
    if cfg:
        r = cfg.get("_project_root")
        if r is not None:
            return Path(r).resolve()
        pr = cfg.get("project_root")
        if pr:
            return Path(pr).resolve()
    return _REPO_ROOT


def default_alpha_path() -> Path:
    y = _REPO_ROOT / "PROXY" / "config" / "agriculture" / "alpha.config.yaml"
    if y.is_file():
        return y
    return _REPO_ROOT / "PROXY" / "config" / "agriculture" / "alpha.config.json"


def load_alpha_config(path: Path | None = None) -> dict[str, Any]:
    p = path
    if p is None:
        env = os.environ.get("K_AGRICULTURE_ALPHA_CONFIG", "").strip()
        p = Path(env) if env else default_alpha_path()
    if not p.is_file():
        return {}
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_default_agriculture_dict() -> dict[str, Any]:
    base = _REPO_ROOT / "PROXY" / "config" / "agriculture"
    y = base / "defaults.yaml"
    if y.is_file():
        return load_yaml_mapping(y)
    legacy = base / "defaults.json"
    if legacy.is_file():
        return json.loads(legacy.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"K_Agriculture defaults not found: {y} (or legacy defaults.json)")


def load_agriculture_config(
    path: Path | None = None,
    *,
    alpha_path: Path | None = None,
) -> dict[str, Any]:
    """Load legacy-style merged json (file + alpha); used if ``K_AGRICULTURE_CONFIG`` is set."""
    p = path
    if p is None:
        env = os.environ.get("K_AGRICULTURE_CONFIG", os.environ.get("AGRICULTURE_CONFIG", "")).strip()
        p = Path(env) if env else None
    if p is None or not p.is_file():
        cfg = load_default_agriculture_dict()
    else:
        if p.suffix.lower() in (".yaml", ".yml"):
            cfg = load_yaml_mapping(p)
        else:
            cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["alpha"] = load_alpha_config(alpha_path)
    return cfg
