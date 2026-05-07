"""Derived attribute columns (offroad_family, shipping_family, waste_family, aviation_family) from OSM tags."""

from __future__ import annotations

import json
from typing import Any


def waste_family(tags: dict[str, str]) -> str:
    if tags.get("landuse") == "landfill":
        return "landfill"
    if tags.get("man_made") == "wastewater_plant":
        return "wastewater_plant"
    am = tags.get("amenity")
    if am == "waste_disposal":
        return "amenity_waste_disposal"
    if am == "recycling":
        return "amenity_recycling"
    return "other"


def offroad_family(
    tags: dict[str, str],
    *,
    rail_line: frozenset[str],
    landuse_poly: frozenset[str],
    rail_depot: frozenset[str],
    landuse_rail: frozenset[str],
    mm_pipe: frozenset[str],
    mm_works: frozenset[str],
    industrial_depot: frozenset[str],
) -> str:
    rw = tags.get("railway")
    if rw in rail_line:
        return f"railway_{rw}"
    if tags.get("landuse") in landuse_rail:
        return "landuse_railway"
    if rw in rail_depot:
        return f"railway_{rw}"
    if tags.get("man_made") in mm_pipe:
        return "man_made_pipeline"
    if tags.get("man_made") in mm_works:
        return "man_made_works"
    if tags.get("industrial") in industrial_depot:
        return "industrial_depot"
    lu = tags.get("landuse")
    if lu in landuse_poly:
        return f"landuse_{lu}"
    return "other"


def aviation_family(tags: dict[str, str]) -> str:
    """Classify airport-footprint polygons for GNFR H / proxy bucketing."""
    if tags.get("aeroway") == "aerodrome":
        return "aerodrome"
    if tags.get("landuse") == "aerodrome":
        return "landuse_aerodrome"
    if tags.get("military") == "airfield":
        return "military_airfield"
    return "other"


def shipping_family(tags: dict[str, str], priority: str) -> str:
    if tags.get("landuse") == "port":
        return "landuse_port"
    if tags.get("harbour") == "yes":
        return "harbour_yes"
    if tags.get("natural") == "harbour":
        return "natural_harbour"
    if tags.get("industrial") == "shipyard":
        return "industrial_shipyard"
    if tags.get("landuse") == "industrial":
        return "landuse_industrial"
    mm = tags.get("man_made")
    if mm == "pier":
        return "man_made_pier"
    if mm == "quay":
        return "man_made_quay"
    if mm == "breakwater":
        return "man_made_breakwater"
    return f"other_{priority}"


def base_row(
    tags: dict[str, str],
    osm_element_id: int,
    osm_element_type: str,
    geometry: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "osm_element_id": osm_element_id,
        "osm_element_type": osm_element_type,
        "osm_tags": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        "name": tags.get("name"),
        "geometry": geometry,
    }
    if extra:
        row.update(extra)
    return row
