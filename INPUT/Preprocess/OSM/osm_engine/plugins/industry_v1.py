#!/usr/bin/env python3
"""
Extract OSM features for industrial / extractive / heavy manufacturing proxy layers.

Follows the pipeline pattern of build_osm_energy_polygons.py / build_osm_offroad_layers.py:
bbox extract (osmium), optional tags-filter, pyosmium SimpleHandler, clip to NUTS,
dedupe by OSM id, write GeoPackage (EPSG:3035).

Element kinds (see rule sets K_WR, K_WRN, …):
  node       — OSM node (point)
  way_open   — open way → LineString
  area_way   — Area object from closed way (polygon)
  area_rel   — Area object from multipolygon relation
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
from shapely import from_wkb
from shapely.geometry import Point as ShpPoint

try:
    import osmium
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'osmium' Python package (pyosmium) is required. "
        "Install with: pip install osmium geopandas shapely pyproj"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PBF = PROJECT_ROOT / "INPUT" / "Preprocess" / "OSM" / "_source" / "europe-latest.osm.pbf"
DEFAULT_NUTS = PROJECT_ROOT / "INPUT" / "Proxy" / "Boundaries" / "NUTS_RG_20M_2021_3035.gpkg"
DEFAULT_OUT = PROJECT_ROOT / "INPUT" / "Proxy" / "OSM" / "industry_layers.gpkg"
TARGET_CRS = "EPSG:3035"

MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL = 250 * 1024 * 1024

# Rule geometry sets (Overpass-style [W,R,N] → our kinds)
K_WR = frozenset({"way_open", "area_way", "area_rel"})
K_WRN = frozenset({"node", "way_open", "area_way", "area_rel"})
K_WN = frozenset({"node", "way_open", "area_way"})
K_W = frozenset({"way_open", "area_way", "area_rel"})
K_N = frozenset({"node"})

REFINERY_NAME_RE = re.compile(r"refinery|raffinerie|raffineria|raffinería", re.I)
BUILDING_REFINERY_NAME_RE = re.compile(r"refinery|raffinerie|raffineria", re.I)

COMMON_TAG_COLUMNS: tuple[str, ...] = (
    "landuse",
    "industrial",
    "man_made",
    "power",
    "building",
    "amenity",
    "craft",
    "product",
    "content",
    "plant_source",
    "generator_source",
    "name",
)

# Broad osmium tags-filter (prefetch); --prefilter-tags
OSMIUM_TAG_FILTERS: tuple[str, ...] = (
    "nwr/landuse=industrial",
    "nwr/landuse=port",
    "nwr/landuse=depot",
    "nwr/landuse=railway",
    "nwr/landuse=quarry",
    "nwr/landuse=brownfield",
    "nwr/industrial",
    "nwr/man_made",
    "nwr/power=plant",
    "nwr/power=generator",
    "nwr/building",
    "nwr/amenity=fuel_depot",
    "nwr/craft=oil_mill",
    "nwr/product",
    "nwr/content",
)


def _wkb_to_bytes(wkb: bytes | str | memoryview) -> bytes:
    if isinstance(wkb, bytes):
        return wkb
    if isinstance(wkb, memoryview):
        return bytes(wkb)
    if isinstance(wkb, str):
        return wkb.encode("latin-1")
    return bytes(wkb)


def _tags_to_dict(taglist: Any) -> dict[str, str]:
    return {t.k: t.v for t in taglist}


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


def _ok(kind: str, kinds: frozenset[str]) -> bool:
    return kind in kinds


def classify_industrial_layer(tags: dict[str, str], elem_kind: str) -> str | None:
    """
    First matching rule wins (most specific rules must appear before broad landuse=industrial).
    """
    t = tags

    # --- Specific landuse + name ---
    if (
        _ok(elem_kind, K_WR)
        and t.get("landuse") == "industrial"
        and REFINERY_NAME_RE.search(t.get("name") or "")
    ):
        return "landuse_industrial_name_refinery_regex"

    # --- industrial=* oil/petroleum block [W,R] ---
    for val, lid in (
        ("refinery", "industrial_refinery"),
        ("oil", "industrial_oil"),
        ("fuel", "industrial_fuel"),
        ("petroleum", "industrial_petroleum"),
        ("oil_mill", "industrial_oil_mill"),
        ("oil_terminal", "industrial_oil_terminal"),
        ("fuel_depot", "industrial_fuel_depot"),
    ):
        if _ok(elem_kind, K_WR) and t.get("industrial") == val:
            return lid

    # --- man_made=works + product (subset; full list below in loop) ---
    works_products_wrn = (
        "oil",
        "fuel",
        "petroleum",
        "diesel",
        "gasoline",
        "kerosene",
        "bitumen",
        "lubricant",
        "cement",
        "lime",
        "glass",
        "ceramic",
        "brick",
        "gravel",
        "sand",
        "stone",
        "aggregate",
        "asphalt",
        "concrete",
        "gypsum",
        "chemical",
        "chemicals",
        "pharmaceutical",
        "fertilizer",
        "ammonia",
        "chlorine",
        "polymer",
        "plastic",
        "acid",
        "paint",
        "gas",
        "nitrogen",
        "oxygen",
        "steel",
        "iron",
        "aluminium",
        "aluminum",
        "copper",
        "zinc",
        "lead",
        "metal",
        "wire",
    )
    if t.get("man_made") == "works":
        prod = t.get("product")
        if prod in works_products_wrn and _ok(elem_kind, K_WRN):
            return f"works_product_{prod}"

    # --- storage_tank + content [W,N] ---
    if t.get("man_made") == "storage_tank":
        ct = t.get("content")
        tank_wn = (
            "oil",
            "fuel",
            "diesel",
            "gasoline",
            "petroleum",
            "crude",
            "chemical",
            "acid",
            "ammonia",
            "chlorine",
        )
        if ct in tank_wn and _ok(elem_kind, K_WN):
            return f"storage_tank_content_{ct}"

    # --- petroleum / oil well [W,N] ---
    if _ok(elem_kind, K_WN):
        if t.get("man_made") == "petroleum_well":
            return "man_made_petroleum_well"
        if t.get("man_made") == "oil_well":
            return "man_made_oil_well"

    # --- amenity / craft ---
    if _ok(elem_kind, K_WN) and t.get("amenity") == "fuel_depot":
        return "amenity_fuel_depot"
    if _ok(elem_kind, K_WN) and t.get("craft") == "oil_mill":
        return "craft_oil_mill"

    # --- building + name refinery [W] ---
    if (
        _ok(elem_kind, K_W)
        and t.get("building")
        and BUILDING_REFINERY_NAME_RE.search(t.get("name") or "")
    ):
        return "building_name_refinery_regex"

    # --- landuse drivers [W,R] ---
    for lu, lid in (
        ("port", "landuse_port"),
        ("depot", "landuse_depot"),
        ("railway", "landuse_railway"),
        ("quarry", "landuse_quarry"),
        ("brownfield", "landuse_brownfield"),
    ):
        if _ok(elem_kind, K_WR) and t.get("landuse") == lu:
            return lid

    # --- industrial= large enumerations [W,R,N] ---
    ind_wrn = (
        "factory",
        "manufacture",
        "manufacturing",
        "engineering",
        "machine_shop",
        "workshop",
        "metal_works",
        "textile",
        "textile_mill",
        "food",
        "food_processing",
        "brewery",
        "distillery",
        "sugar",
        "tobacco",
        "wood",
        "sawmill",
        "lumber",
        "furniture",
        "rubber",
        "plastics",
        "printing",
        "electronics",
        "electrical",
        "vehicle",
        "shipyard",
        "boatyard",
        "aircraft",
        "railway",
        "recycling",
        "scrap",
        "slaughterhouse",
        "dairy",
        "bakery",
        "fish_processing",
        "leather",
        "hemp",
        "packaging",
        "warehouse",
        "logistics",
        "depot",
        "quarry",
        "mine",
        "cement",
        "cement_plant",
        "lime",
        "lime_kiln",
        "glass",
        "glassworks",
        "ceramic",
        "ceramics",
        "pottery",
        "brick",
        "brickworks",
        "tileworks",
        "sand",
        "gravel",
        "aggregate",
        "stone",
        "mineral",
        "gypsum",
        "chalk",
        "clay",
        "salt",
        "asphalt",
        "asphalt_plant",
        "concrete",
        "concrete_plant",
        "ready_mix_concrete",
        "chemical",
        "chemical_plant",
        "chemicals",
        "petrochemical",
        "pharmaceutical",
        "pharmaceuticals",
        "fertilizer",
        "fertiliser",
        "agrochemical",
        "pesticide",
        "paint",
        "paints",
        "varnish",
        "polymer",
        "resin",
        "acid",
        "chlorine",
        "ammonia",
        "detergent",
        "soap",
        "cosmetics",
        "explosives",
        "pyrotechnics",
        "gas_plant",
        "oxygen_plant",
        "nitrogen_plant",
        "carbon_black",
        "solvent",
        "adhesive",
        "ink",
        "steel",
        "steelworks",
        "iron",
        "iron_works",
        "ironworks",
        "metal",
        "metalworks",
        "foundry",
        "smelting",
        "smelter",
        "blast_furnace",
        "aluminium",
        "aluminum",
        "aluminium_smelter",
        "copper",
        "zinc",
        "lead",
        "nickel",
        "titanium",
        "magnesium",
        "precious_metal",
        "gold",
        "silver",
        "rolling_mill",
        "wire_drawing",
        "forging",
        "casting",
        "galvanizing",
        "electroplating",
        "scrap_metal",
    )
    for ind in ind_wrn:
        if _ok(elem_kind, K_WRN) and t.get("industrial") == ind:
            return f"industrial_{ind}"

    # --- man_made plants [W,R,N] ---
    if _ok(elem_kind, K_WRN):
        if t.get("man_made") == "factory":
            return "man_made_factory"
        if t.get("man_made") == "kiln":
            return "man_made_kiln"

    # --- smokestack / chimney / mineshaft / adit [N] ---
    if _ok(elem_kind, K_N):
        mm = t.get("man_made")
        if mm == "chimney":
            return "man_made_chimney"
        if mm == "smokestack":
            return "man_made_smokestack"
        if mm == "mineshaft":
            return "man_made_mineshaft"
        if mm == "adit":
            return "man_made_adit"

    # --- cooling_tower, gasometer, boiler_house [W,N]; blast_furnace [W,N] ---
    if _ok(elem_kind, K_WN):
        mm = t.get("man_made")
        if mm == "cooling_tower":
            return "man_made_cooling_tower"
        if mm == "gasometer":
            return "man_made_gasometer"
        if mm == "boiler_house":
            return "man_made_boiler_house"
        if mm == "blast_furnace":
            return "man_made_blast_furnace"

    # --- buildings industrial [W] ---
    if _ok(elem_kind, K_W):
        b = t.get("building")
        if b in (
            "industrial",
            "factory",
            "warehouse",
            "manufacture",
            "works",
            "workshop",
            "hangar",
        ):
            return f"building_{b}"

    # --- power plant / generator [W,R] / [W,N] ---
    if t.get("power") == "plant":
        ps = t.get("plant:source") or t.get("plant_source")
        if ps in ("gas", "oil", "biomass", "coal") and _ok(elem_kind, K_WR):
            return f"power_plant_plant_source_{ps}"
    if t.get("power") == "generator":
        gs = t.get("generator:source") or t.get("generator_source")
        if gs in ("gas", "oil", "biomass") and _ok(elem_kind, K_WN):
            return f"power_generator_generator_source_{gs}"

    # --- generic man_made=works [W,R,N] (after product-specific) ---
    if _ok(elem_kind, K_WRN) and t.get("man_made") == "works":
        return "man_made_works_generic"

    # --- PRIMARY landuse=industrial (broad) [W,R] ---
    if _ok(elem_kind, K_WR) and t.get("landuse") == "industrial":
        return "landuse_industrial"

    return None


def _resolve_osmium_exe(cli: str | None) -> str | None:
    if cli:
        p = Path(cli).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--osmium does not exist: {p}")
        return str(p)
    exe = shutil.which("osmium")
    return str(Path(exe).resolve()) if exe else None


def _run_osmium(osmium_exe: str, *tail: str) -> None:
    r = subprocess.run([osmium_exe, *tail], check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"osmium failed ({r.returncode}): {' '.join(tail[:8])}\n"
            "See osmium error output printed above."
        )


def _tags_filter(osmium_exe: str, source: Path, out: Path, filters: Iterable[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    filt = tuple(filters)
    last: RuntimeError | None = None
    for want_progress in (True, False):
        try:
            if want_progress:
                _run_osmium(
                    osmium_exe,
                    "tags-filter",
                    "--progress",
                    "--overwrite",
                    "-o",
                    str(out),
                    str(source),
                    *filt,
                )
            else:
                _run_osmium(
                    osmium_exe,
                    "tags-filter",
                    "--no-progress",
                    "--overwrite",
                    "-o",
                    str(out),
                    str(source),
                    *filt,
                )
            if not want_progress:
                warnings.warn(
                    "osmium tags-filter succeeded without progress bar (first attempt failed).",
                    stacklevel=2,
                )
            return
        except RuntimeError as e:
            last = e
            if want_progress:
                warnings.warn(
                    "osmium tags-filter failed with --progress; retrying with --no-progress.",
                    stacklevel=2,
                )
    raise RuntimeError(
        "osmium tags-filter failed twice (with and without progress bar)."
    ) from last


def _extract_bbox(osmium_exe: str, bbox: str, source: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    last: RuntimeError | None = None
    for want_progress in (True, False):
        try:
            if want_progress:
                _run_osmium(
                    osmium_exe,
                    "extract",
                    "--progress",
                    "--overwrite",
                    "-b",
                    bbox,
                    "-o",
                    str(out),
                    str(source),
                )
            else:
                _run_osmium(
                    osmium_exe,
                    "extract",
                    "--no-progress",
                    "--overwrite",
                    "-b",
                    bbox,
                    "-o",
                    str(out),
                    str(source),
                )
            if not want_progress:
                warnings.warn(
                    "osmium extract succeeded without progress bar (first attempt failed).",
                    stacklevel=2,
                )
            return
        except RuntimeError as e:
            last = e
            if want_progress:
                warnings.warn(
                    "osmium extract failed with --progress; retrying with --no-progress.",
                    stacklevel=2,
                )
    raise RuntimeError(
        "osmium extract failed twice (with and without progress bar)."
    ) from last


def _load_boundary(nuts_path: Path, cntr_code: str | None) -> tuple[gpd.GeoDataFrame, int]:
    gdf = gpd.read_file(nuts_path)
    if cntr_code is not None:
        gdf = gdf[gdf["CNTR_CODE"] == cntr_code.upper()].copy()
        if gdf.empty:
            raise SystemExit(f"No features for CNTR_CODE={cntr_code!r} in {nuts_path}")
    n_features = len(gdf)
    return gdf.dissolve(), n_features


def _bbox_str_wgs84(boundary_3035: gpd.GeoDataFrame) -> str:
    b = boundary_3035.to_crs(4326).total_bounds
    west, south, east, north = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return f"{west},{south},{east},{north}"


def _make_row(
    tags: dict[str, str],
    *,
    layer: str,
    osm_element_id: int,
    osm_element_type: str,
    geometry: Any,
) -> dict[str, Any]:
    ex = _expand_tags(tags)
    row: dict[str, Any] = {
        "osm_element_id": osm_element_id,
        "osm_element_type": osm_element_type,
        "industrial_layer": layer,
        "osm_tags": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        "geometry": geometry,
    }
    for k in COMMON_TAG_COLUMNS:
        row[k] = ex.get(k)
    return row


class IndustrialOSMCollector(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.wkb_factory = osmium.geom.WKBFactory()
        self.points: list[dict[str, Any]] = []
        self.lines: list[dict[str, Any]] = []
        self.polygons: list[dict[str, Any]] = []

    def node(self, n: osmium.osm.Node) -> None:
        tags = _tags_to_dict(n.tags)
        layer = classify_industrial_layer(tags, "node")
        if not layer:
            return
        try:
            loc = n.location
            if not loc.valid():
                return
            geom = ShpPoint(float(loc.lon), float(loc.lat))
        except Exception:
            return
        self.points.append(_make_row(tags, layer=layer, osm_element_id=n.id, osm_element_type="node", geometry=geom))

    def way(self, w: osmium.osm.Way) -> None:
        tags = _tags_to_dict(w.tags)
        layer = classify_industrial_layer(tags, "way_open")
        if not layer:
            return
        try:
            wkb = self.wkb_factory.create_linestring(w)
            geom = from_wkb(_wkb_to_bytes(wkb))
        except Exception:
            return
        if geom.geom_type not in ("LineString", "MultiLineString"):
            return
        self.lines.append(_make_row(tags, layer=layer, osm_element_id=w.id, osm_element_type="way", geometry=geom))

    def area(self, a: osmium.osm.Area) -> None:
        tags = _tags_to_dict(a.tags)
        elem_kind = "area_way" if a.from_way() else "area_rel"
        layer = classify_industrial_layer(tags, elem_kind)
        if not layer:
            return
        try:
            wkb = self.wkb_factory.create_multipolygon(a)
            geom = from_wkb(_wkb_to_bytes(wkb))
        except Exception:
            return
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            return
        oid = a.orig_id()
        et = "way" if a.from_way() else "relation"
        self.polygons.append(_make_row(tags, layer=layer, osm_element_id=oid, osm_element_type=et, geometry=geom))


def _empty_gdf_wgs84() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs="EPSG:4326")})


def _dedupe_osm_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    k = gdf["osm_element_type"].astype(str) + ":" + gdf["osm_element_id"].astype(str)
    return gdf.loc[~k.duplicated(keep="first")].copy()


def _clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_wgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gt = gdf.geometry.geom_type
    poly_m = gt.isin(["Polygon", "MultiPolygon"])
    if poly_m.any():
        gdf.loc[poly_m, "geometry"] = gdf.loc[poly_m, "geometry"].buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]
    gdf = gpd.clip(gdf, boundary_wgs)
    if gdf.empty:
        return gdf
    return gdf.to_crs(TARGET_CRS)


def _write_gpkg(out: Path, layers: Iterable[tuple[str, gpd.GeoDataFrame]]) -> None:
    nonempty = [(n, g) for n, g in layers if not g.empty]
    if not nonempty:
        raise SystemExit("All layers empty after processing; check PBF, NUTS mask, and tag rules.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass
    name0, gdf0 = nonempty[0]
    gdf0.to_file(out, layer=name0, driver="GPKG", mode="w")
    for name, gdf in nonempty[1:]:
        gdf.to_file(out, layer=name, driver="GPKG", mode="a")
    print(f"Wrote {out} ({len(nonempty)} layers, crs={TARGET_CRS})")
    for name, gdf in nonempty:
        print(f"  {name}: {len(gdf)} features", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OSM industrial / extractive layers → GeoPackage (EPSG:3035), clipped to NUTS."
    )
    p.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    p.add_argument("--nuts", type=Path, default=DEFAULT_NUTS)
    p.add_argument("--cntr-code", type=str, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--osmium", type=str, default=None)
    p.add_argument("--no-bbox-extract", action="store_true")
    p.add_argument(
        "--prefilter-tags",
        action="store_true",
        help="Run osmium tags-filter before pyosmium (smaller file; may drop some multipolygons).",
    )
    p.add_argument("--allow-large-pbf-without-osmium", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable pyosmium tqdm bar (osmium-tool still prints its own --progress when used).",
    )
    return p.parse_args()


def _extract_object_total_from_fileinfo_json(data: Any) -> int | None:
    """Best-effort nodes+ways+relations from ``osmium fileinfo -e -j`` JSON."""
    if isinstance(data, dict):
        if all(k in data for k in ("nodes", "ways", "relations")):
            try:
                return int(data["nodes"]) + int(data["ways"]) + int(data["relations"])
            except (TypeError, ValueError):
                pass
        for v in data.values():
            t = _extract_object_total_from_fileinfo_json(v)
            if t is not None:
                return t
    if isinstance(data, list):
        for x in data:
            t = _extract_object_total_from_fileinfo_json(x)
            if t is not None:
                return t
    return None


def _estimate_pbf_object_count(osmium_exe: str, pbf: Path) -> int | None:
    """Slow on huge PBF; used only as tqdm total hint for pyosmium stage."""
    try:
        r = subprocess.run(
            [osmium_exe, "fileinfo", "-e", "-j", str(pbf)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        data = json.loads(r.stdout)
        return _extract_object_total_from_fileinfo_json(data)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError, ValueError):
        return None


class IndustrialOSMCollectorWithProgress(IndustrialOSMCollector):
    """Counts OSM objects during scan for tqdm (does not change classification)."""

    def __init__(self, pbar: Any, *, tick_every: int = 50_000) -> None:
        super().__init__()
        self._pbar = pbar
        self._tick_every = tick_every
        self.objects_seen = 0

    def _tick(self) -> None:
        self.objects_seen += 1
        if self.objects_seen % self._tick_every == 0:
            self._pbar.update(self._tick_every)

    def node(self, n: osmium.osm.Node) -> None:  # type: ignore[name-defined]
        self._tick()
        super().node(n)

    def way(self, w: osmium.osm.Way) -> None:  # type: ignore[name-defined]
        self._tick()
        super().way(w)

    def relation(self, r: osmium.osm.Relation) -> None:  # type: ignore[name-defined]
        # Progress only: SimpleHandler defines no relation(); IndustrialOSMCollector ignores relations.
        self._tick()

    def area(self, a: osmium.osm.Area) -> None:  # type: ignore[name-defined]
        self._tick()
        super().area(a)


def main() -> None:
    args = _parse_args()
    pbf = args.pbf.expanduser().resolve()
    if not pbf.is_file():
        raise SystemExit(f"PBF not found: {pbf}")
    nuts = args.nuts.expanduser().resolve()
    if not nuts.is_file():
        raise SystemExit(f"Boundary not found: {nuts}")

    boundary, _n = _load_boundary(nuts, args.cntr_code)
    bbox_wgs84 = _bbox_str_wgs84(boundary)

    osmium_exe = _resolve_osmium_exe(args.osmium)
    work_pbf = pbf
    temp_root: Path | None = None

    if (
        not osmium_exe
        and pbf.stat().st_size > MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL
        and not args.allow_large_pbf_without_osmium
    ):
        raise SystemExit(
            f"PBF is large ({pbf.stat().st_size / (1024**3):.1f} GiB) and osmium-tool was not found.\n"
            "Install conda-forge::osmium-tool or pass --allow-large-pbf-without-osmium"
        )

    show_py_progress = not args.no_progress

    n_pipeline_steps = (
        int(bool(osmium_exe and not args.no_bbox_extract))
        + int(bool(osmium_exe and args.prefilter_tags))
        + 2
    )
    _step_n = 0

    def _stage(msg: str) -> None:
        nonlocal _step_n
        _step_n += 1
        print(f"[{_step_n}/{n_pipeline_steps}] {msg}", flush=True)

    try:
        if osmium_exe and not args.no_bbox_extract:
            _stage(f"Bounding-box extract (osmium, bbox={bbox_wgs84}) …")
            if temp_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="osm_industrial_"))
            bbox_pbf = temp_root / "bbox_extract.osm.pbf"
            _extract_bbox(osmium_exe, bbox_wgs84, pbf, bbox_pbf)
            work_pbf = bbox_pbf
            print("    extract finished.", flush=True)

        if osmium_exe and args.prefilter_tags:
            _stage("Tags prefilter (osmium tags-filter) …")
            if temp_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="osm_industrial_"))
            filtered = temp_root / "tags_filtered.osm.pbf"
            _tags_filter(osmium_exe, work_pbf, filtered, OSMIUM_TAG_FILTERS)
            work_pbf = filtered
            print("    tags-filter finished.", flush=True)
        elif not osmium_exe:
            warnings.warn(
                "osmium-tool not found: full PBF read may be very slow / high RAM.",
                stacklevel=1,
            )

        _stage(
            "Parse OSM with industrial rules (pyosmium — often the slow step; "
            "progress counts objects scanned; install tqdm for a bar: pip install tqdm) …"
        )

        tqdm_cls = None
        if show_py_progress:
            try:
                from tqdm import tqdm as tqdm_cls  # type: ignore[no-redef,attr-defined]
            except ImportError:
                tqdm_cls = None

        total_hint: int | None = None
        if tqdm_cls is not None and osmium_exe:
            total_hint = _estimate_pbf_object_count(osmium_exe, work_pbf)

        pbar = None
        col: IndustrialOSMCollector
        if tqdm_cls is not None:
            pbar = tqdm_cls(
                total=total_hint,
                unit="obj",
                desc="pyosmium scan",
                smoothing=0.05,
                mininterval=0.5,
            )
            col = IndustrialOSMCollectorWithProgress(pbar)
        else:
            col = IndustrialOSMCollector()

        try:
            col.apply_file(str(work_pbf), locations=True, idx="flex_mem")
        finally:
            if pbar is not None and isinstance(col, IndustrialOSMCollectorWithProgress):
                rem = col.objects_seen % col._tick_every
                if rem:
                    pbar.update(rem)
                pbar.close()

        _stage("Clip to NUTS, dedupe, write GeoPackage …")

        boundary_wgs = boundary.to_crs(4326)

        def to_gdf(rows: list[dict[str, Any]]) -> gpd.GeoDataFrame:
            if not rows:
                return _empty_gdf_wgs84()
            return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

        gdf_pt = _dedupe_osm_id(_clip_to_boundary(to_gdf(col.points), boundary_wgs))
        gdf_ln = _dedupe_osm_id(_clip_to_boundary(to_gdf(col.lines), boundary_wgs))
        gdf_pg = _dedupe_osm_id(_clip_to_boundary(to_gdf(col.polygons), boundary_wgs))

        out = args.out.expanduser().resolve()
        _write_gpkg(
            out,
            [
                ("osm_industrial_polygons", gdf_pg),
                ("osm_industrial_lines", gdf_ln),
                ("osm_industrial_points", gdf_pt),
            ],
        )
        print(f"    wrote {len([x for x in [gdf_pg, gdf_ln, gdf_pt] if not x.empty])}/3 nonempty layers.", flush=True)
    finally:
        if temp_root is not None:
            if args.keep_temp:
                print(f"Kept temp dir: {temp_root}", flush=True)
            else:
                shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
