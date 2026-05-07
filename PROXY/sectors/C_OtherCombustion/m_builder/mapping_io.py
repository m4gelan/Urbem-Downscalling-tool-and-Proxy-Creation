"""Load GAINS→class mapping sidecar (YAML or JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .sidecar_io import load_sidecar_dict


def load_gains_mapping(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_sidecar_dict(path)
    rules = data.get("rules")
    if not isinstance(rules, list):
        rules = []
    clean_rules = [r for r in rules if isinstance(r, dict) and "class" in r]
    hints = data.get("emep_fuel_hints") or []
    if not isinstance(hints, list):
        hints = []
    clean_hints = [h for h in hints if isinstance(h, dict)]
    return clean_rules, clean_hints
