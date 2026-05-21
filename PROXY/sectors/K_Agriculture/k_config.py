"""Project root and config helpers for K_Agriculture (replaces top-level ``Agriculture.config``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
    """Load optional merged YAML/JSON; ``alpha_path`` is ignored (CEIP alpha only)."""
    _ = alpha_path
    p = path
    if p is None:
        env = os.environ.get("K_AGRICULTURE_CONFIG", os.environ.get("AGRICULTURE_CONFIG", "")).strip()
        p = Path(env) if env else None
    if p is None or not p.is_file():
        return load_default_agriculture_dict()
    if p.suffix.lower() in (".yaml", ".yml"):
        return load_yaml_mapping(p)
    return json.loads(p.read_text(encoding="utf-8"))
