from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
from shapely import from_wkb
from shapely.geometry import Point as ShpPoint

from . import common
from . import log
from . import pyosmium_io
from .predicates import match_classify_rule


def _expand_tags(tags: dict[str, str]) -> dict[str, str]:
    """Expose plant:source / generator:source as flat keys for GeoPackage columns."""
    out = dict(tags)
    ps = tags.get("plant:source") or tags.get("plant_source")
    if ps:
        out["plant_source"] = ps
    gs = tags.get("generator:source") or tags.get("generator_source")
    if gs:
        out["generator_source"] = gs
    return out


def first_classify_value(
    tags: dict[str, str],
    elem_kind: str,
    rules: list[dict[str, Any]],
) -> str | None:
    """Return industrial_layer (or other output column value) from first matching classify rule."""
    for r in rules:
        if match_classify_rule(tags, elem_kind, r):
            return str(r["value"])
    return None


def _layer_for_elem_kind(sector: dict[str, Any], elem_kind: str) -> str:
    """Map element kind to output GPKG layer name from sector layer_map."""
    lm = sector.get("layer_map") or {}
    if elem_kind == "node":
        return str(lm.get("node", "osm_industrial_points"))
    if elem_kind == "way_open":
        return str(lm.get("way_open", "osm_industrial_lines"))
    return str(lm.get("area", "osm_industrial_polygons"))


class ClassifyCollector(osmium.SimpleHandler):
    """pyosmium handler: first-match classify_rules → buckets by output layer."""

    def __init__(
        self,
        sector: dict[str, Any],
        *,
        rules: list[dict[str, Any]],
        store_osm_tags: bool,
        row_columns: list[str],
        output_column: str,
    ) -> None:
        super().__init__()
        self.sector = sector
        self.rules = rules
        self.store_osm_tags = store_osm_tags
        self.row_columns = row_columns
        self.output_column = output_column
        self.wkb_factory = osmium.geom.WKBFactory()
        self.buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._n_obj = 0

    def _make_row(
        self,
        tags: dict[str, str],
        value: str,
        osm_id: int,
        osm_etype: str,
        geometry: Any,
    ) -> dict[str, Any]:
        """Build one output row for a classified feature."""
        ex = _expand_tags(tags)
        row: dict[str, Any] = {
            "osm_element_id": osm_id,
            "osm_element_type": osm_etype,
            self.output_column: value,
            "geometry": geometry,
        }
        if self.store_osm_tags:
            row["osm_tags"] = json.dumps(tags, ensure_ascii=False, sort_keys=True)
        for k in self.row_columns:
            row[k] = ex.get(k)
        return row

    def _emit(self, tags: dict[str, str], elem_kind: str, osm_id: int, osm_etype: str, geometry: Any) -> None:
        """Classify tags and append row to the appropriate layer bucket."""
        value = first_classify_value(tags, elem_kind, self.rules)
        if not value:
            return
        layer = _layer_for_elem_kind(self.sector, elem_kind)
        self.buckets[layer].append(self._make_row(tags, value, osm_id, osm_etype, geometry))

    def node(self, n: osmium.osm.Node) -> None:
        """Handle OSM nodes."""
        self._n_obj += 1
        tags = common.tags_to_dict(n.tags)
        try:
            loc = n.location
            if not loc.valid():
                return
            geom = ShpPoint(float(loc.lon), float(loc.lat))
        except Exception:
            return
        self._emit(tags, "node", n.id, "node", geom)

    def way(self, w: osmium.osm.Way) -> None:
        """Handle open ways as lines."""
        self._n_obj += 1
        tags = common.tags_to_dict(w.tags)
        try:
            wkb = self.wkb_factory.create_linestring(w)
            geom = from_wkb(common.wkb_to_bytes(wkb))
        except Exception:
            return
        if geom.geom_type not in ("LineString", "MultiLineString"):
            return
        self._emit(tags, "way_open", w.id, "way", geom)

    def area(self, a: osmium.osm.Area) -> None:
        """Handle area objects (closed ways and multipolygon relations)."""
        self._n_obj += 1
        tags = common.tags_to_dict(a.tags)
        elem_kind = "area_way" if a.from_way() else "area_rel"
        try:
            wkb = self.wkb_factory.create_multipolygon(a)
            geom = from_wkb(common.wkb_to_bytes(wkb))
        except Exception:
            return
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            return
        oid = a.orig_id()
        et = "way" if a.from_way() else "relation"
        self._emit(tags, elem_kind, oid, et, geom)

    def object_count(self) -> int:
        """Return number of OSM objects visited."""
        return self._n_obj

    def kept_count(self) -> int:
        """Return total classified features kept across layers."""
        return sum(len(v) for v in self.buckets.values())


def run_classify_sector(
    sector_id: str,
    sector: dict[str, Any],
    *,
    work_pbf: Path,
    boundary_wgs: gpd.GeoDataFrame,
    out: Path,
    min_m2: float,
    pyosmium_idx: str = "flex_mem",
    osmium_exe: str | None = None,
    defaults: dict[str, Any] | None = None,
    sector_entry: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Parse work PBF with ClassifyCollector and write classified layers."""
    rules = sector.get("classify_rules")
    if not isinstance(rules, list) or not rules:
        raise SystemExit(f"Sector {sector_id!r}: mode classify requires non-empty classify_rules")

    store_osm_tags = sector.get("store_osm_tags") is True
    row_columns = list(sector.get("row_tags") or [])
    output_column = str(sector.get("classify_output_column", "industrial_layer"))
    layer_order = list(
        sector.get("layer_order")
        or [
            sector.get("layer_map", {}).get("area", "osm_industrial_polygons"),
            sector.get("layer_map", {}).get("way_open", "osm_industrial_lines"),
            sector.get("layer_map", {}).get("node", "osm_industrial_points"),
        ]
    )

    t0 = log.Timer()
    log.sector_info(sector_id, f"parse start {work_pbf.name} idx={pyosmium_idx}")
    col = ClassifyCollector(
        sector,
        rules=rules,
        store_osm_tags=bool(store_osm_tags),
        row_columns=row_columns,
        output_column=output_column,
    )
    pyosmium_io.apply_file(
        col,
        work_pbf,
        sector_id=sector_id,
        idx=pyosmium_idx,
        osmium_exe=osmium_exe,
        defaults=defaults or {},
        sector_entry=sector_entry or {},
    )
    log.sector_info(
        sector_id,
        f"parse done objects={col.object_count()} kept={col.kept_count()} ({log.format_duration(t0.elapsed())})",
    )

    from . import pipeline

    layers = pipeline.postprocess_layers(
        dict(col.buckets),
        layer_order=layer_order,
        boundary_wgs=boundary_wgs,
        min_m2=min_m2,
    )
    return pipeline.write_sector_gpkg(out, layers, sector_id=sector_id)
