from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REFERENCE_POLLUTANTS = ("PM10", "NOx", "SOx", "NMVOC")


def reference_pollutants_from_cfg(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("reference_pollutants")
    if raw:
        return [str(x).strip() for x in raw if str(x).strip()]
    if cfg.get("reference_pollutant"):
        return [str(cfg["reference_pollutant"]).strip()]
    return list(REFERENCE_POLLUTANTS)


def load_reference_pollutants(repo_root: Path) -> list[str]:
    for name in ("weight_sensitivity_prong_a_w.yaml", "weight_sensitivity_prong_a.yaml"):
        path = repo_root / "proxy" / "config" / name
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            if isinstance(doc, dict):
                return reference_pollutants_from_cfg(doc)
    return list(REFERENCE_POLLUTANTS)
