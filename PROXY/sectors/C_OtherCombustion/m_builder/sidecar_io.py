"""Load GNFR C science sidecars from YAML or JSON (suffix decides parser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_sidecar_dict(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    return data if isinstance(data, dict) else {}
