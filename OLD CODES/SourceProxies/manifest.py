"""JSON manifest next to GeoTIFF outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    out = dict(payload)
    out["created_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
