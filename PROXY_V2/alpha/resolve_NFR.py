from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def normalize_inventory_sector(value: Any) -> str:
    """Canonical NFR / inventory sector token (alphanumeric only, upper-case)."""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


@dataclass(frozen=True)
class SectorGroupResolution:
    """Sector block from ``alpha_methods.yaml``: ordered group names and NFR (inventory) code → group."""

    group_order: tuple[str, ...]
    code_norm_to_group: dict[str, str]

    def group_for_sector_norm(self, sector_norm: str) -> str | None:
        return self.code_norm_to_group.get(sector_norm)


def resolve_sector_groups_from_block(sector_block: dict[str, Any]) -> SectorGroupResolution:
    """
    Build group order and inventory-sector membership from a sector YAML block ``groups:`` mapping.

    Keys are the authoritative group names. Each value is a list of NFR-style codes.
    Codes must not appear in more than one group.
    """
    raw = sector_block.get("groups")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("sector block must contain non-empty 'groups' mapping")

    group_order: list[str] = []
    code_norm_to_group: dict[str, str] = {}
    for group_name, codes in raw.items():
        g = str(group_name).strip()
        if not g:
            raise ValueError("empty group name in sector groups")
        if g in group_order:
            raise ValueError(f"duplicate group key {g!r}")
        group_order.append(g)
        if not isinstance(codes, list) or not codes:
            raise ValueError(f"group {g!r} must be a non-empty list of NFR codes")
        for c in codes:
            tok = normalize_inventory_sector(c)
            if not tok:
                continue
            if tok in code_norm_to_group:
                raise ValueError(
                    f"NFR code {tok!r} appears in both {code_norm_to_group[tok]!r} and {g!r}"
                )
            code_norm_to_group[tok] = g

    return SectorGroupResolution(
        group_order=tuple(group_order),
        code_norm_to_group=dict(code_norm_to_group),
    )
