"""osmium SimpleHandler driven by ``osm_schema.yaml`` rules (nodes / lines / areas)."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import osmium
from shapely import from_wkb
from shapely.geometry import Point as ShpPoint

from . import common
from . import families
from .predicates import match_tags


def _when_ok(rule: dict[str, Any], ctx: dict[str, Any]) -> bool:
    w = rule.get("when")
    if w is None:
        return True
    if isinstance(w, str):
        return bool(ctx.get(w))
    return True


def _apply_area_first(rules: list[dict[str, Any]], tags: dict[str, str]) -> list[dict[str, Any]]:
    for r in rules:
        if match_tags(tags, r.get("match")):
            return [r]
    return []


def _line_strategy_first(rules: list[dict[str, Any]], tags: dict[str, str]) -> list[dict[str, Any]]:
    for r in rules:
        if match_tags(tags, r.get("match")):
            return [r]
    return []


class RulesCollector(osmium.SimpleHandler):
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

    def _row(
        self,
        tags: dict[str, str],
        osm_id: int,
        osm_etype: str,
        geometry: Any,
        rule: dict[str, Any],
    ) -> dict[str, Any]:
        extra = dict(rule.get("extra_fields") or {})
        row: dict[str, Any] = {
            "osm_element_id": osm_id,
            "osm_element_type": osm_etype,
            "osm_tags": json.dumps(tags, ensure_ascii=False, sort_keys=True),
            "geometry": geometry,
        }
        for k in self.sector.get("row_tags", []):
            if k not in extra:
                row[k] = tags.get(k)
        row.update(extra)
        aug = self.sector.get("augment_family")
        if aug == "waste":
            row["waste_family"] = families.waste_family(tags)
        elif aug == "offroad" and self.offroad_sets:
            s = self.offroad_sets
            row["offroad_family"] = families.offroad_family(
                tags,
                rail_line=s["railway_line_values"],
                landuse_poly=s["landuse_polygon_values"],
                rail_depot=s["railway_depot_yard"],
                landuse_rail=s["landuse_rail"],
                mm_pipe=s["man_made_line_or_area"],
                mm_works=s["man_made_works"],
                industrial_depot=s["industrial_value"],
            )
            for k2 in (
                "name",
                "landuse",
                "railway",
                "man_made",
                "industrial",
                "electrified",
                "substance",
                "substation",
                "pipeline",
            ):
                row[k2] = tags.get(k2)
        elif aug == "shipping":
            pr = str(extra.get("shipping_priority", "high"))
            row["shipping_priority"] = pr
            row["shipping_family"] = families.shipping_family(tags, pr)
            for k2 in (
                "name",
                "landuse",
                "harbour",
                "natural",
                "industrial",
                "man_made",
            ):
                row[k2] = tags.get(k2)
        elif aug == "solvent":
            for k2 in (
                "name",
                "landuse",
                "building",
                "highway",
                "aeroway",
                "shop",
                "amenity",
                "man_made",
            ):
                if k2 not in row:
                    row[k2] = tags.get(k2)
        elif aug == "aviation":
            row["aviation_family"] = families.aviation_family(tags)
            for k2 in ("name", "aeroway", "iata", "icao", "operator", "military", "landuse"):
                row[k2] = tags.get(k2)
        return row

    def _emit_node(self, n: osmium.osm.Node, tags: dict[str, str]) -> None:
        rules = self.sector.get("rules", {}).get("nodes") or []
        to_apply: list[dict[str, Any]]
        if self._node_mode == "first":
            to_apply = []
            for r in rules:
                if not _when_ok(r, self.ctx):
                    continue
                if match_tags(tags, r.get("match")):
                    to_apply = [r]
                    break
        else:
            to_apply = [r for r in rules if _when_ok(r, self.ctx) and match_tags(tags, r.get("match"))]

        for r in to_apply:
            for layer in r.get("layers", []):
                try:
                    loc = n.location
                    if not loc.valid():
                        return
                    geom = ShpPoint(float(loc.lon), float(loc.lat))
                except Exception:
                    return
                self.buckets[layer].append(
                    self._row(tags, n.id, "node", geom, r),
                )

    def _line_rules(self) -> list[dict[str, Any]]:
        return self.sector.get("rules", {}).get("lines") or []

    def _area_rules(self) -> list[dict[str, Any]]:
        return self.sector.get("rules", {}).get("areas") or []

    def node(self, n: osmium.osm.Node) -> None:
        self._emit_node(n, common.tags_to_dict(n.tags))

    def way(self, w: osmium.osm.Way) -> None:
        tags = common.tags_to_dict(w.tags)
        rules = [r for r in self._line_rules() if _when_ok(r, self.ctx)]
        to_apply: list[dict[str, Any]]
        if self._line_mode == "first":
            to_apply = _line_strategy_first(rules, tags)
        else:
            to_apply = [r for r in rules if match_tags(tags, r.get("match"))]

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
        tags = common.tags_to_dict(a.tags)
        rules = [r for r in self._area_rules() if _when_ok(r, self.ctx)]
        to_apply: list[dict[str, Any]]
        if self._area_mode == "multi":
            to_apply = [r for r in rules if match_tags(tags, r.get("match"))]
        else:
            to_apply = _apply_area_first(rules, tags)

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
