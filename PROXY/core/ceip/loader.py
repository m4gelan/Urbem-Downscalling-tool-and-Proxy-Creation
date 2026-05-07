from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_INDEX_REL = Path("PROXY/config/ceip/index.yaml")

# Default CEIP ``groups:`` keys for B_Industry / D_Fugitive-style profiles. Sectors may
# override with ``ceip_group_order`` / ``group_order`` in YAML (e.g. three named legs).
DEFAULT_GNFR_GROUP_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4")

# Old defaults -> canonical paths under config/ceip/ (user sector YAMLs may still list these).
LEGACY_PROFILE_PATHS: dict[str, str] = {
    "PROXY/config/industry/ceip_groups.yaml": "PROXY/config/ceip/profiles/industry_groups.yaml",
    "PROXY/config/fugitive/ceip_groups.yaml": "PROXY/config/ceip/profiles/fugitive_groups.yaml",
    "PROXY/config/waste/ceip_families.yaml": "PROXY/config/ceip/profiles/waste_families.yaml",
    "PROXY/config/solvents/ceip_subsectors.yaml": "PROXY/config/ceip/profiles/solvents_subsectors.yaml",
}


def remap_legacy_ceip_relpath(path_str: str) -> str:
    """Map deprecated profile paths to config/ceip/profiles/ equivalents."""
    s = (path_str or "").replace("\\", "/").strip()
    return LEGACY_PROFILE_PATHS.get(s, s)


@lru_cache(maxsize=1)
def _load_index(root: str) -> dict[str, Any]:
    p = Path(root) / _INDEX_REL
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def clear_ceip_index_cache() -> None:
    _load_index.cache_clear()


def _sectors_block(root: Path) -> dict[str, Any]:
    return (_load_index(str(root.resolve())) or {}).get("sectors") or {}


def default_ceip_profile_relpath(root: Path, sector_id: str, resource: str) -> str:
    """
    Project-relative path from ``config/ceip/index.yaml`` for a sector.

    ``resource`` is a key in the sector block, e.g. ``groups_yaml``,
    ``families_yaml``, ``subsectors_yaml``.
    """
    entry = _sectors_block(root).get(sector_id) or {}
    p = entry.get(resource)
    if not p or not isinstance(p, str):
        raise KeyError(
            f"ceip index: sector {sector_id!r} has no {resource!r} "
            f"(see PROXY/config/ceip/index.yaml)"
        )
    return p


def shared_pollutant_aliases_relpath(root: Path) -> str | None:
    """Project-relative path from index ``shared.pollutant_aliases_yaml``, or None."""
    shared = (_load_index(str(root.resolve())) or {}).get("shared") or {}
    p = shared.get("pollutant_aliases_yaml")
    return str(p) if isinstance(p, str) and p.strip() else None
