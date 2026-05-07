"""Offroad OSM proxy-rule helpers.

Inputs are the loaded path/config mapping, specifically the `proxy_rules`
section. Outputs are normalized filter sets used by Offroad railway-line
processing to exclude non-relevant or inactive OSM rail features.
"""

from __future__ import annotations

from typing import Any


def osm_railway_line_filter_sets(path_cfg: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """
    Return ``(bad_line_types, lifecycle_disallow_values)`` for OSM rail line filtering.

    Values are lower-case strings, compared to ``railway`` / tag values on features.
    When YAML omits lists, defaults align with ``osm_engine/osm_schema.yaml`` offroad rail rules.
    """
    block = (path_cfg.get("proxy_rules") or {}).get("osm_railway") or {}
    bad = [str(x).strip().lower() for x in (block.get("bad_line_types") or [])]
    life = [str(x).strip().lower() for x in (block.get("lifecycle_disallow_railway_tag_values") or [])]
    if not bad:
        bad = []
    if not life:
        life = ["abandoned", "disused", "razed", "proposed", "construction", "dismantled"]
    return frozenset(bad), frozenset(life)


__all__ = ["osm_railway_line_filter_sets"]
