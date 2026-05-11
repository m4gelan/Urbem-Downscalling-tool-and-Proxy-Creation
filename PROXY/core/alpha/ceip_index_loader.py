from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from PROXY.core.dataloaders import resolve_path

logger = logging.getLogger(__name__)

_INDEX_REL = Path("PROXY/config/ceip/index.yaml")


def _package_ceip_index_path() -> Path:
    """``PROXY/config/ceip/index.yaml`` next to the installed ``PROXY`` package (sibling of ``core/``)."""
    return Path(__file__).resolve().parents[2] / "config" / "ceip" / "index.yaml"


def _candidate_ceip_index_paths(root: Path) -> list[Path]:
    """Prefer workspace-relative index; fall back if ``root`` does not contain ``PROXY/config/...``."""
    primary = Path(root).resolve() / _INDEX_REL
    fb = _package_ceip_index_path()
    if fb.resolve() == primary.resolve():
        return [primary]
    return [primary, fb]

# Default CEIP ``groups:`` keys for B_Industry / D_Fugitive-style profiles. Sectors may
# override with ``ceip_group_order`` / ``group_order`` in YAML (e.g. three named legs).
DEFAULT_GNFR_GROUP_ORDER: tuple[str, ...] = ("G1", "G2", "G3", "G4")

# Old defaults -> canonical paths under config/ceip/ (user sector YAMLs may still list these).
LEGACY_PROFILE_PATHS: dict[str, str] = {
    "PROXY/config/industry/ceip_groups.yaml": "PROXY/config/ceip/profiles/B_Industry_groups.yaml",
    "PROXY/config/fugitive/ceip_groups.yaml": "PROXY/config/ceip/profiles/D_Fugitive_groups.yaml",
    "PROXY/config/waste/ceip_families.yaml": "PROXY/config/ceip/profiles/J_Waste_groups.yaml",
    "PROXY/config/solvents/ceip_subsectors.yaml": "PROXY/config/ceip/profiles/E_Solvents_groups.yaml",
    # Legacy monolithic filenames -> groups file (rules come from index + merge loader).
    "PROXY/config/ceip/profiles/industry_groups.yaml": "PROXY/config/ceip/profiles/B_Industry_groups.yaml",
    "PROXY/config/ceip/profiles/fugitive_groups.yaml": "PROXY/config/ceip/profiles/D_Fugitive_groups.yaml",
    "PROXY/config/ceip/profiles/waste_families.yaml": "PROXY/config/ceip/profiles/J_Waste_groups.yaml",
    "PROXY/config/ceip/profiles/solvents_subsectors.yaml": "PROXY/config/ceip/profiles/E_Solvents_groups.yaml",
    "PROXY/config/ceip/profiles/waste_pipeline.yaml": "PROXY/config/ceip/profiles/J_Waste_rules.yaml",
    "PROXY/config/ceip/profiles/solvents_pipeline.yaml": "PROXY/config/ceip/profiles/E_Solvents_rules.yaml",
}


def remap_legacy_ceip_relpath(path_str: str) -> str:
    """Map deprecated profile paths to config/ceip/profiles/ equivalents."""
    s = (path_str or "").replace("\\", "/").strip()
    return LEGACY_PROFILE_PATHS.get(s, s)


@lru_cache(maxsize=32)
def _load_index(root: str) -> dict[str, Any]:
    candidates = _candidate_ceip_index_paths(Path(root))
    chosen: Path | None = None
    for i, p in enumerate(candidates):
        if p.is_file():
            chosen = p
            if i > 0:
                logger.warning(
                    "CEIP index: %s not found; using %s (fix project root / paths so the primary exists).",
                    candidates[0],
                    p,
                )
            break
    if chosen is None:
        logger.warning(
            "CEIP index: no index.yaml found (tried %s); CEIP sector paths will fail.",
            ", ".join(str(x) for x in candidates),
        )
        return {}
    with chosen.open(encoding="utf-8") as f:
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


def load_merged_ceip_profile_for_pipeline_paths(
    root: Path,
    paths_cfg: dict[str, Any],
    *,
    profile_sector_id: str | None = None,
) -> dict[str, Any]:
    """
    Load merged CEIP profile from ``paths_cfg`` (``ceip_groups_yaml`` + optional
    ``ceip_rules_yaml``), mirroring :func:`run_gnfr_group_pipeline` resolution.

    Used by Folium context rebuilds and other tools that need the same merged
    ``groups`` dict as the GNFR pipelines.
    """
    from .ceip_profile_merge import load_merged_ceip_profile

    gy = paths_cfg.get("ceip_groups_yaml")
    if gy is None:
        raise KeyError("paths.ceip_groups_yaml")
    gpath = Path(gy) if isinstance(gy, Path) else Path(str(gy))
    if not gpath.is_absolute():
        gpath = resolve_path(root, remap_legacy_ceip_relpath(str(gy)))

    rules_resolved: Path | None = None
    ry = paths_cfg.get("ceip_rules_yaml")
    if ry:
        rp = Path(ry) if isinstance(ry, Path) else Path(str(ry))
        rules_resolved = rp if rp.is_absolute() else resolve_path(root, remap_legacy_ceip_relpath(str(ry)))
    elif profile_sector_id:
        try:
            rr = default_ceip_profile_relpath(root, profile_sector_id, "rules_yaml")
            cand = resolve_path(root, Path(rr))
            if cand.is_file():
                rules_resolved = cand
        except KeyError:
            pass

    return load_merged_ceip_profile(
        gpath,
        rules_resolved if rules_resolved is not None and rules_resolved.is_file() else None,
    )
