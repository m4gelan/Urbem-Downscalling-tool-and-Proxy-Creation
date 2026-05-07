"""Load agriculture pipeline configuration (paths, run) and alpha table (separate JSON)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .core import project_root_from_here


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "agriculture.config.json"


def default_alpha_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "alpha.config.json"


def load_alpha_config(path: Path | None = None) -> dict[str, Any]:
    p = path
    if p is None:
        env = os.environ.get("AGRICULTURE_ALPHA_CONFIG", "").strip()
        p = Path(env) if env else default_alpha_config_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_agriculture_config(path: Path | None = None, alpha_path: Path | None = None) -> dict[str, Any]:
    p = path or default_config_path()
    if not p.is_file():
        raise FileNotFoundError(f"Agriculture config not found: {p}")
    cfg: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    cfg["alpha"] = load_alpha_config(alpha_path)
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    pr = cfg.get("project_root")
    if pr:
        return Path(pr).resolve()
    return project_root_from_here()
