#!/usr/bin/env python3
"""
Interactive Folium map of Greece OSM extracts (roads + landuse/buildings PBF).

Reads ``OSM_roads_Greece.osm.pbf`` (layer ``lines``) and ``OSM_landuse_buildings_Greece.osm.pbf``
(layer ``multipolygons``). Roads are split into toggleable groups by **highway** class;
polygons into **building** and **landuse** groups (bucketed for a readable layer list).

By default only the **first** ``--max-roads`` / ``--max-polygons`` rows of each layer are
read (PBF order, not a spatial sample). If the map looks sparse, raise those caps or use
``0`` for **no limit** (entire layer: slow, heavy RAM, very large HTML — browser may
struggle). Geometry is simplified by default to limit file size.

Usage (from project root):
  python Solvents/Auxiliaries/osm_greece_folium_map.py
  python Solvents/Auxiliaries/osm_greece_folium_map.py --out-html Solvents/outputs/osm_greece.html
  python Solvents/Auxiliaries/osm_greece_folium_map.py --max-roads 0 --max-polygons 0 --simplify-m 4

Requires: geopandas, folium (and a GDAL build with OSM / PBF read support in the geopandas stack).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROADS = PROJECT_ROOT / "data" / "OSM" / "OSM_roads_Greece.osm.pbf"
DEFAULT_LANDUSE = PROJECT_ROOT / "data" / "OSM" / "OSM_landuse_buildings_Greece.osm.pbf"
DEFAULT_OUT = PROJECT_ROOT / "Solvents" / "outputs" / "osm_greece_layers.html"

GREECE_CENTER = (38.25, 24.5)
DEFAULT_ZOOM = 7

# (layer name, tuple of highway=* values)
ROAD_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Roads — motorways", ("motorway", "motorway_link")),
    ("Roads — trunk / primary", ("trunk", "trunk_link", "primary", "primary_link")),
    ("Roads — secondary / tertiary", ("secondary", "secondary_link", "tertiary", "tertiary_link")),
    ("Roads — minor (residential, living, unclassified…)", ("residential", "living_street", "unclassified", "road", "service")),
    ("Roads — tracks / paths / cycle / foot", ("track", "path", "footway", "cycleway", "pedestrian", "steps", "bridleway")),
]

BUILDING_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Buildings — housing (house, detached, terrace…)", ("house", "detached", "terrace", "bungalow", "semidetached_house", "hut", "cabin", "dormitory")),
    ("Buildings — apartments / flats", ("apartments", "residential")),
    ("Buildings — commercial / retail / offices", ("commercial", "retail", "office", "warehouse", "industrial", "supermarket", "kiosk")),
    ("Buildings — public (school, hospital, church…)", ("school", "university", "hospital", "church", "public", "civic", "government", "train_station")),
    ("Buildings — farm / agricultural", ("farm", "barn", "greenhouse", "silo", "stable")),
]

LANDUSE_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Landuse — farmland / meadow / orchard", ("farmland", "farmyard", "meadow", "orchard", "vineyard", "plant_nursery")),
    ("Landuse — forest / vegetation", ("forest", "wood", "grass")),
    ("Landuse — built-up (residential, commercial…)", ("residential", "commercial", "industrial", "retail", "construction")),
    ("Landuse — transport / infrastructure", ("railway", "military", "port", "quarry", "landfill")),
    ("Landuse — water / wetlands", ("basin", "reservoir", "salt_pond")),
]


def _stable_color(key: str) -> str:
    if not key:
        return "#888888"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"#{h[0:6]}"


def _norm_tag(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s


def _building_bucket(b: str) -> str | None:
    b = _norm_tag(b).lower()
    if not b:
        return None
    for name, tags in BUILDING_BUCKETS:
        if b in tags:
            return name
    if b == "yes":
        return "Buildings — generic (building=yes)"
    return "Buildings — other types"


def _landuse_bucket(lu: str) -> str | None:
    lu = _norm_tag(lu).lower()
    if not lu:
        return None
    for name, tags in LANDUSE_BUCKETS:
        if lu in tags:
            return name
    return "Landuse — other"


def _road_bucket(hw: str) -> str:
    hw = _norm_tag(hw).lower()
    if not hw:
        return "Roads — other / unknown"
    for name, tags in ROAD_BUCKETS:
        if hw in tags:
            return name
    return "Roads — other / unknown"


def _read_osm_lines(path: Path, max_rows: int | None) -> "geopandas.GeoDataFrame":
    import geopandas as gpd

    kw: dict = {"layer": "lines"}
    if max_rows is not None:
        kw["rows"] = max_rows
    gdf = gpd.read_file(path, **kw)
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    return gdf


def _read_osm_multipolygons(path: Path, max_rows: int | None) -> "geopandas.GeoDataFrame":
    import geopandas as gpd

    kw: dict = {"layer": "multipolygons"}
    if max_rows is not None:
        kw["rows"] = max_rows
    gdf = gpd.read_file(path, **kw)
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    return gdf


def _add_geojson_group(
    fmap,
    gdf: "geopandas.GeoDataFrame",
    name: str,
    *,
    stroke_color: str,
    fill_color: str | None,
    fill_opacity: float,
    weight: int,
    fields: list[str],
    show: bool,
) -> None:
    import folium

    if gdf.empty:
        return
    fg = folium.FeatureGroup(name=name, show=show)
    tip_fields = [f for f in fields if f in gdf.columns]
    kw: dict = {
        "style_function": lambda _f, sc=stroke_color, fc=fill_color, fo=fill_opacity, w=weight: {
            "color": sc,
            "weight": w,
            "opacity": 0.85,
            "fillColor": fc or sc,
            "fillOpacity": fo if fc else 0.0,
        },
        "highlight_function": lambda _f, w=weight, fo=fill_opacity: {
            "weight": w + 2,
            "fillOpacity": min(0.55, fo + 0.15),
        },
    }
    if tip_fields:
        kw["tooltip"] = folium.GeoJsonTooltip(fields=tip_fields, aliases=tip_fields, sticky=True)
    folium.GeoJson(gdf, **kw).add_to(fg)
    fg.add_to(fmap)


def build_map(
    roads_pbf: Path,
    landuse_pbf: Path,
    out_html: Path,
    *,
    max_roads: int | None,
    max_polygons: int | None,
    simplify_m: float | None,
    zoom_start: int,
) -> None:
    import folium

    if max_roads is None:
        print("Loading roads: no row limit (full lines layer).", flush=True)
    if max_polygons is None:
        print("Loading multipolygons: no row limit (full layer).", flush=True)
    roads = _read_osm_lines(roads_pbf, max_roads)
    polys = _read_osm_multipolygons(landuse_pbf, max_polygons)

    if simplify_m and simplify_m > 0:
        tol = simplify_m / 111_000.0
        roads = roads.copy()
        polys = polys.copy()
        roads["geometry"] = roads.geometry.simplify(tol, preserve_topology=True)
        polys["geometry"] = polys.geometry.simplify(tol, preserve_topology=True)

    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    polys = polys[polys.geometry.notna() & ~polys.geometry.is_empty]

    if "highway" in roads.columns:
        roads["highway"] = roads["highway"].map(_norm_tag)
    else:
        roads["highway"] = ""

    for col in ("building", "landuse"):
        if col in polys.columns:
            polys[col] = polys[col].map(_norm_tag)
        else:
            polys[col] = ""

    roads["road_layer"] = roads["highway"].map(_road_bucket)
    buildings = polys[polys["building"] != ""].copy()
    buildings["building_layer"] = buildings["building"].map(_building_bucket)
    landonly = polys[(polys["building"] == "") & (polys["landuse"] != "")].copy()
    landonly["landuse_layer"] = landonly["landuse"].map(_landuse_bucket)

    fmap = folium.Map(location=list(GREECE_CENTER), zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.TileLayer("CartoDB positron", name="Basemap (light)", control=True).add_to(fmap)

    road_layer_names = list(dict.fromkeys(roads["road_layer"]))
    for i, layer_name in enumerate(sorted(road_layer_names)):
        sub = roads[roads["road_layer"] == layer_name]
        if sub.empty:
            continue
        c = _stable_color(layer_name)
        _add_geojson_group(
            fmap,
            sub,
            layer_name,
            stroke_color=c,
            fill_color=None,
            fill_opacity=0.0,
            weight=2 if "motorway" in layer_name else 1,
            fields=["highway", "name", "osm_id"],
            show=i < 3,
        )

    if not buildings.empty:
        for i, layer_name in enumerate(sorted(buildings["building_layer"].unique())):
            sub = buildings[buildings["building_layer"] == layer_name]
            if sub.empty:
                continue
            c = _stable_color(layer_name)
            _add_geojson_group(
                fmap,
                sub,
                layer_name,
                stroke_color="#333333",
                fill_color=c,
                fill_opacity=0.35,
                weight=1,
                fields=["building", "name", "osm_id"],
                show=False,
            )

    if not landonly.empty:
        for i, layer_name in enumerate(sorted(landonly["landuse_layer"].unique())):
            sub = landonly[landonly["landuse_layer"] == layer_name]
            if sub.empty:
                continue
            c = _stable_color(layer_name + "_fill")
            _add_geojson_group(
                fmap,
                sub,
                layer_name,
                stroke_color=c,
                fill_color=c,
                fill_opacity=0.25,
                weight=1,
                fields=["landuse", "name", "osm_id"],
                show=False,
            )

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    ap = argparse.ArgumentParser(description="Folium map: Greece OSM roads + landuse/buildings layers.")
    ap.add_argument("--roads", type=Path, default=DEFAULT_ROADS, help="Roads .osm.pbf")
    ap.add_argument("--landuse", type=Path, default=DEFAULT_LANDUSE, help="Landuse/buildings .osm.pbf")
    ap.add_argument("--out-html", type=Path, default=DEFAULT_OUT, help="Output HTML path")
    ap.add_argument(
        "--max-roads",
        type=int,
        default=120_000,
        help="Max line features to read from the PBF (first rows in file order). Use 0 for no limit.",
    )
    ap.add_argument(
        "--max-polygons",
        type=int,
        default=120_000,
        help="Max multipolygon features to read. Use 0 for no limit (all buildings/landuse in the extract).",
    )
    ap.add_argument(
        "--simplify-m",
        type=float,
        default=12.0,
        help="Simplify geometries (meters, ~degrees/111e3). Use 0 to disable.",
    )
    ap.add_argument("--zoom", type=int, default=DEFAULT_ZOOM, help="Initial zoom")
    args = ap.parse_args()

    roads = args.roads.expanduser().resolve()
    land = args.landuse.expanduser().resolve()
    out = args.out_html.expanduser().resolve()

    if not roads.is_file():
        raise SystemExit(f"Roads file not found: {roads}")
    if not land.is_file():
        raise SystemExit(f"Landuse/buildings file not found: {land}")

    sim = None if args.simplify_m <= 0 else float(args.simplify_m)
    max_r = None if args.max_roads == 0 else int(args.max_roads)
    max_p = None if args.max_polygons == 0 else int(args.max_polygons)
    if max_r is None or max_p is None:
        print(
            "WARNING: no row cap — loading entire layer(s). Expect long runtime, high memory, "
            "and a large HTML file; your browser may freeze or fail to open it.",
            flush=True,
        )
    build_map(
        roads,
        land,
        out,
        max_roads=max_r,
        max_polygons=max_p,
        simplify_m=sim,
        zoom_start=int(args.zoom),
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
