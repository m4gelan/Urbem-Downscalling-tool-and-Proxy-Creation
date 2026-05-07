from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root(from_path: Path | None = None) -> Path:
    base = from_path or Path(__file__)
    resolved = base.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in [resolved, *resolved.parents]:
        if (candidate / "PROXY").is_dir():
            return candidate
    # Fallback for in-package calls.
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    return data


def deep_resolve_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {k: deep_resolve_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [deep_resolve_paths(v, root) for v in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return resolve_path(root, value)
    return value


@dataclass(frozen=True)
class PathConfig:
    path: Path
    data: dict[str, Any]
    resolved: dict[str, Any]

    def require(self, *keys: str) -> Any:
        cur: Any = self.resolved
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                joined = ".".join(keys)
                raise KeyError(f"Missing config key: {joined}")
            cur = cur[key]
        return cur


def load_path_config(path: Path) -> PathConfig:
    root = project_root(path)
    data = load_yaml(path)
    resolved = deep_resolve_paths(data, root)
    return PathConfig(path=path, data=data, resolved=resolved)

