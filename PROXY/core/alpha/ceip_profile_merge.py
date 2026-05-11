"""Merge CEIP ``*_groups.yaml`` + ``*_rules.yaml`` into one document (same shape as legacy monolithic YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Any, overlay: Any) -> Any:
    """Recursive dict merge: overlay wins on leaf keys; dict values merge recursively."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        out: dict[str, Any] = dict(base)
        for k, v in overlay.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return overlay if overlay is not None else base


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CEIP profile YAML not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def load_merged_ceip_profile(
    groups_path: Path,
    rules_path: Path | None,
) -> dict[str, Any]:
    """
    Load ``groups_path`` and optionally deep-merge ``rules_path``.

    When ``rules_path`` is None or missing on disk, returns the groups document only
    (supports legacy single-file profiles that already contain full ``groups`` specs).
    """
    doc = load_yaml_file(groups_path)
    if rules_path is None or not rules_path.is_file():
        return doc
    rules_doc = load_yaml_file(rules_path)
    return deep_merge(doc, rules_doc)


__all__ = [
    "deep_merge",
    "load_merged_ceip_profile",
    "load_yaml_file",
]
