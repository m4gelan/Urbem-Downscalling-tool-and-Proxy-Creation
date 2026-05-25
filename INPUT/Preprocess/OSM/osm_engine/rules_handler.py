from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import osmium
from shapely import from_wkb
from shapely.geometry import Point as ShpPoint

from . import augment
from . import common
from . import log
from .predicates import build_rule_index, first_matching_rule, matching_rules, match_tags


def _when_ok(rule: dict[str, Any], ctx: dict[str, Any]) -> bool:
    """Return True if rule when-clause passes given run context."""
    w = rule.get("when")
    if w is None:
        return True
    if isinstance(w, str):
        return bool(ctx.get(w))
    return True


def _active_rules(rules: list[dict[str, Any]], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rules whose when-clause passes."""
    return [r for r in rules if _when_ok(r, ctx)]


def _relevant_keys(rule_index: dict[str, set[int]]) -> frozenset[str]:
    """Tag keys that appear in indexed rules."""
    return frozenset(rule_index.keys())


class RulesCollector(osmium.SimpleHandler):
    """pyosmium handler driven by osm_schema.yaml rules (nodes / lines / areas)."""

    def __init__(
        self,
        sector: dict[str, Any],
        *,
        ctx: dict[str, Any],
        offroad_sets: dict[str, frozenset[str]] | None = None,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.ctx = ctx
        self.offroad_sets = offroad_sets or {}
        self.wkb_factory = osmium.geom.WKBFactory()
        self.buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        st = sector.get("strategies") or {}
        self._area_mode = st.get("areas", "first")
        self._line_mode = st.get("lines", "first")
        self._node_mode = st.get("nodes", "first")
        self._store_osm_tags = sector.get("store_osm_tags", log.debug_enabled())

        raw = sector.get("rules") or {}
        self._line_rules = _active_rules(raw.get("lines") or [], ctx)
        self._area_rules = _active_rules(raw.get("areas") or [], ctx)
        self._node_rules = _active_rules(raw.get("nodes") or [], ctx)
        self._line_index = build_rule_index(self._line_rules)
        self._area_index = build_rule_index(self._area_rules)
        self._node_index = build_rule_index(self._node_rules)
        self._line_keys = _relevant_keys(self._line_index)
        self._area_keys = _relevant_keys(self._area_index)
        self._node_keys = _relevant_keys(self._node_index)

    def _row(
        self,
        tags: dict[str, str],
        osm_id: int,
        osm_etype: str,
        geometry: Any,
        rule: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one output row from tags, geometry, and matched rule."""
        extra = dict(rule.get("extra_fields") or {})
        row: dict[str, Any] = {
            "osm_element_id": osm_id,
            "osm_element_type": osm_etype,
            "geometry": geometry,
        }
        if self._store_osm_tags:
            row["osm_tags"] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        for k in self.sector.get("row_tags", []):
            if k not in extra:
                row[k] = tags.get(k)
        row.update(extra)
        augment.apply_row_augment(self.sector, tags, row, self.ctx)
        return row

    def kept_count(self) -> int:
        """Return total features stored across all layers."""
        return sum(len(v) for v in self.buckets.values())

    def layer_counts(self) -> dict[str, int]:
        """Return feature count per layer name."""
        return {k: len(v) for k, v in self.buckets.items()}

    def _emit_node(self, n: osmium.osm.Node, tags: dict[str, str]) -> None:
        """Match node rules and append rows."""
        if self._node_keys and not any(k in tags for k in self._node_keys):
            return

        if self._node_mode == "first":
            rule = first_matching_rule(tags, self._node_rules, self._node_index)
            to_apply = [rule] if rule else []
        else:
            to_apply = matching_rules(tags, self._node_rules, self._node_index)

        for r in to_apply:
            for layer in r.get("layers", []):
                try:
                    loc = n.location
                    if not loc.valid():
                        return
                    geom = ShpPoint(float(loc.lon), float(loc.lat))
                except Exception:
                    return
                self.buckets[layer].append(self._row(tags, n.id, "node", geom, r))

    def node(self, n: osmium.osm.Node) -> None:
        """Handle OSM nodes."""
        self._emit_node(n, common.tags_to_dict(n.tags))

    def way(self, w: osmium.osm.Way) -> None:
        """Handle OSM ways as lines."""
        tags = common.tags_to_dict(w.tags)
        if self._line_keys and not any(k in tags for k in self._line_keys):
            return

        if self._line_mode == "first":
            rule = first_matching_rule(tags, self._line_rules, self._line_index)
            to_apply = [rule] if rule else []
        else:
            to_apply = matching_rules(tags, self._line_rules, self._line_index)

        for r in to_apply:
            try:
                wkb = self.wkb_factory.create_linestring(w)
                geom = from_wkb(common.wkb_to_bytes(wkb))
            except Exception:
                continue
            if geom.geom_type not in ("LineString", "MultiLineString"):
                continue
            for layer in r.get("layers", []):
                self.buckets[layer].append(self._row(tags, w.id, "way", geom, r))

    def area(self, a: osmium.osm.Area) -> None:
        """Handle OSM areas (polygons)."""
        tags = common.tags_to_dict(a.tags)
        if self._area_keys and not any(k in tags for k in self._area_keys):
            return

        if self._area_mode == "multi":
            to_apply = matching_rules(tags, self._area_rules, self._area_index)
        else:
            rule = first_matching_rule(tags, self._area_rules, self._area_index)
            to_apply = [rule] if rule else []

        for r in to_apply:
            try:
                wkb = self.wkb_factory.create_multipolygon(a)
                geom = from_wkb(common.wkb_to_bytes(wkb))
            except Exception:
                continue
            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            oid = a.orig_id()
            et = "way" if a.from_way() else "relation"
            for layer in r.get("layers", []):
                self.buckets[layer].append(self._row(tags, oid, et, geom, r))
