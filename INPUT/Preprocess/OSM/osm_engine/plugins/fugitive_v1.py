#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable

import geopandas as gpd
from shapely import from_wkb


def _wkb_to_bytes(wkb: bytes | str | memoryview) -> bytes:
    """Pyosmium may return WKB as bytes or as a binary str (Windows); Shapely needs bytes."""
    if isinstance(wkb, bytes):
        return wkb
    if isinstance(wkb, memoryview):
        return bytes(wkb)
    if isinstance(wkb, str):
        return wkb.encode("latin-1")
    return bytes(wkb)

try:
    import osmium
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'osmium' Python package (pyosmium) is required. "
        "Install with: pip install osmium geopandas shapely pyproj"
    ) from e

from osm_engine.common import filter_polygon_min_area

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_PBF = PROJECT_ROOT / "INPUT" / "Preprocess" / "OSM" / "_source" / "europe-latest.osm.pbf"
DEFAULT_NUTS = PROJECT_ROOT / "INPUT" / "Proxy" / "Boundaries" / "NUTS_RG_20M_2021_3035.gpkg"
DEFAULT_OUT = PROJECT_ROOT / "INPUT" / "Proxy" / "OSM" / "fugitive_layers.gpkg"
TARGET_CRS = "EPSG:3035"

# Parsing a full regional/planet PBF with pyosmium without a prior bbox/tags extract needs huge RAM.
MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL = 250 * 1024 * 1024

# Optional osmium tags-filter (see --prefilter-tags).
OSMIUM_TAG_FILTERS: tuple[str, ...] = (
    "nwr/landuse=industrial",
    "nwr/landuse=quarry",
    "nwr/landuse=construction",
    "nwr/landuse=brownfield",
    "nwr/landuse=port",
    "nwr/landuse=commercial",
    "nwr/industrial",
    "nwr/man_made=works",
    "nwr/man_made=storage_tank",
    "nwr/man_made=tank",
    "nwr/man_made=pipeline",
    "nwr/man_made=chimney",
    "nwr/man_made=flare"
    "nwr/man_made=petroleum_well"
    "nwr/power=plant",
    "nwr/power=generator",
    "nwr/power=substation",
    "nwr/resource=coal",
    "nwr/resource=oil",
    "nwr/resource=gas",
    "nwr/amenity=fuel",
)

COMMON_TAG_COLUMNS: tuple[str, ...] = (
    "landuse",
    "industrial",
    "man_made",
    "power",
    "amenity",
    "resource",
    "building",
    "name",
)


def _tags_to_dict(taglist: Any) -> dict[str, str]:
    return {t.k: t.v for t in taglist}


def _tag_matches_selection(tags: dict[str, str]) -> bool:
    lu = tags.get("landuse")
    if lu in ("industrial", "quarry", "construction", "brownfield", "port"):
        return True
    if lu == "commercial":
        return True
    if tags.get("industrial"):
        return True
    mm = tags.get("man_made")
    if mm in ("works", "storage_tank", "tank", "pipeline", "chimney", "flare"):
        return True
    pw = tags.get("power")
    if pw in ("plant", "generator", "substation"):
        return True
    res = tags.get("resource")
    if res in ("coal", "oil", "gas"):
        return True
    if tags.get("amenity") == "fuel":
        return True
    ind = tags.get("industrial")
    if ind in ("depot", "fuel_storage"):
        return True
    return False


def _tag_match_families(tags: dict[str, str]) -> str:
    fam: set[str] = set()
    lu = tags.get("landuse")
    if lu in ("industrial", "quarry", "construction", "brownfield", "port"):
        fam.add("landuse_industrial_base")
    if lu == "commercial":
        fam.add("landuse_commercial")
    if tags.get("industrial"):
        fam.add("industrial")
    mm = tags.get("man_made")
    if mm in ("works", "storage_tank", "tank", "pipeline", "chimney", "flare"):
        fam.add("man_made")
    pw = tags.get("power")
    if pw in ("plant", "generator", "substation"):
        fam.add("power")
    res = tags.get("resource")
    if res in ("coal", "oil", "gas"):
        fam.add("resource")
    if tags.get("amenity") == "fuel":
        fam.add("amenity_fuel")
    ind = tags.get("industrial")
    if ind in ("depot", "fuel_storage"):
        fam.add("industrial_depot_storage")
    return ",".join(sorted(fam))


def _resolve_osmium_exe(cli: str | None) -> str | None:
    if cli:
        p = Path(cli).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--osmium does not exist: {p}")
        return str(p)
    exe = shutil.which("osmium")
    return str(Path(exe).resolve()) if exe else None


def _run_osmium(osmium_exe: str, *tail: str) -> None:
    # Inherit stdout/stderr (Windows: piping stderr can trigger bogus printer errors).
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
        "osmium tags-filter failed twice (with and without progress bar).\n"
        "Workarounds: disable tag prefilter if your pipeline allows it, "
        "or run osmium tags-filter manually."
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
        "osmium extract failed twice (with and without progress bar).\n"
        "If you saw 'printer out of paper', stderr progress output often triggers that on Windows.\n"
        "Workarounds: --no-bbox-extract (read full PBF in Python; slow, high RAM); "
        "or run osmium extract manually and pass --pbf to the extracted file."
    ) from last


@dataclass
class AreaParseDebug:
    """Counters for --debug (pyosmium area callback path)."""

    n_callbacks: int = 0
    n_empty_tags: int = 0
    n_tag_no_match: int = 0
    n_geom_error: int = 0
    first_geom_error: str | None = None
    n_wrong_geom_type: int = 0
    first_wrong_geom_type: str | None = None
    n_kept: int = 0
    no_match_samples: list[str] = field(default_factory=list)
    match_samples: list[str] = field(default_factory=list)
    geom_error_samples: list[str] = field(default_factory=list)


def _tag_summary(tags: dict[str, str], max_len: int = 220) -> str:
    s = json.dumps(tags, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


class EnergyAreaCollector(osmium.SimpleHandler):
    def __init__(
        self,
        *,
        debug: AreaParseDebug | None = None,
        debug_sample_limit: int = 5,
    ) -> None:
        super().__init__()
        self.wkb_factory = osmium.geom.WKBFactory()
        self.rows: list[dict[str, Any]] = []
        self._dbg = debug
        self._dbg_limit = max(0, int(debug_sample_limit))

    def _sample(self, bucket: list[str], line: str) -> None:
        if self._dbg_limit and len(bucket) < self._dbg_limit:
            bucket.append(line)

    def area(self, a: osmium.osm.Area) -> None:
        dbg = self._dbg
        if dbg is not None:
            dbg.n_callbacks += 1

        tags = _tags_to_dict(a.tags)
        if not tags and dbg is not None:
            dbg.n_empty_tags += 1
        if not _tag_matches_selection(tags):
            if dbg is not None:
                dbg.n_tag_no_match += 1
                lab = "tags=<empty>" if not tags else _tag_summary(tags)
                self._sample(
                    dbg.no_match_samples,
                    f"orig_id={a.orig_id()} from_way={a.from_way()} {lab}",
                )
            return

        if dbg is not None:
            self._sample(dbg.match_samples, f"orig_id={a.orig_id()} {_tag_summary(tags)}")

        try:
            wkb = self.wkb_factory.create_multipolygon(a)
            geom = from_wkb(_wkb_to_bytes(wkb))
        except Exception as e:
            if dbg is not None:
                dbg.n_geom_error += 1
                if dbg.first_geom_error is None:
                    dbg.first_geom_error = f"{type(e).__name__}: {e}"
                self._sample(
                    dbg.geom_error_samples,
                    f"orig_id={a.orig_id()} {type(e).__name__}: {e!r}",
                )
            return
        gtype = geom.geom_type
        if gtype not in ("Polygon", "MultiPolygon"):
            if dbg is not None:
                dbg.n_wrong_geom_type += 1
                if dbg.first_wrong_geom_type is None:
                    dbg.first_wrong_geom_type = gtype
            return
        oid = a.orig_id()
        from_way = a.from_way()
        etype = "way" if from_way else "relation"
        row = {
            "osm_element_id": oid,
            "osm_element_type": etype,
            "osm_area_id": a.id,
            "osm_tags": json.dumps(tags, ensure_ascii=False, sort_keys=True),
            "tag_match_families": _tag_match_families(tags),
        }
        for k in COMMON_TAG_COLUMNS:
            row[k] = tags.get(k)
        row["geometry"] = geom
        self.rows.append(row)
        if dbg is not None:
            dbg.n_kept += 1


def _print_debug_header(
    *,
    work_pbf: Path,
    bbox_wgs84: str,
    boundary: gpd.GeoDataFrame,
    nuts_n: int,
    cntr_code: str | None,
    prefilter_tags: bool,
    no_bbox_extract: bool,
) -> None:
    print("[debug] pipeline", flush=True)
    print(f"  work_pbf: {work_pbf} ({work_pbf.stat().st_size / (1024**2):.1f} MiB)", flush=True)
    print(f"  bbox WGS84 (west,south,east,north): {bbox_wgs84}", flush=True)
    print(f"  NUTS polygons used for mask (before dissolve): {nuts_n}", flush=True)
    print(f"  --cntr-code: {cntr_code!r}", flush=True)
    if boundary.crs is not None:
        print(f"  boundary CRS: {boundary.crs.to_string()}", flush=True)
    print(f"  prefilter_tags: {prefilter_tags}", flush=True)
    print(f"  no_bbox_extract: {no_bbox_extract}", flush=True)


def _print_area_debug(d: AreaParseDebug) -> None:
    print("[debug] pyosmium area() callback", flush=True)
    print(f"  n_callbacks total: {d.n_callbacks}", flush=True)
    print(f"  n_empty_tags (no OSM tags on Area): {d.n_empty_tags}", flush=True)
    print(f"  n_tag_no_match (selection rule false, includes empty tags): {d.n_tag_no_match}", flush=True)
    print(f"  n_geom_error (WKB / shapely): {d.n_geom_error}", flush=True)
    if d.first_geom_error:
        print(f"  first_geom_error: {d.first_geom_error}", flush=True)
    print(f"  n_wrong_geom_type (not Polygon/MultiPolygon): {d.n_wrong_geom_type}", flush=True)
    if d.first_wrong_geom_type:
        print(f"  first_wrong_geom_type: {d.first_wrong_geom_type}", flush=True)
    print(f"  n_kept (rows before GeoDataFrame): {d.n_kept}", flush=True)
    if d.match_samples:
        print("  sample matched tags:", flush=True)
        for line in d.match_samples:
            print(f"    {line}", flush=True)
    if d.no_match_samples:
        print("  sample no-match (wrong tag combo or empty tags):", flush=True)
        for line in d.no_match_samples:
            print(f"    {line}", flush=True)
    if d.geom_error_samples:
        print("  sample geometry errors:", flush=True)
        for line in d.geom_error_samples:
            print(f"    {line}", flush=True)
    if d.n_callbacks == 0:
        print(
            "  NOTE: zero area() callbacks means pyosmium built no Area objects from this PBF "
            "(incomplete multipolygons, or empty extract).",
            flush=True,
        )


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harmonized OSM energy/fugitive polygons -> GeoPackage (EPSG:3035).")
    p.add_argument("--pbf", type=Path, default=DEFAULT_PBF, help="Source .osm.pbf")
    p.add_argument("--nuts", type=Path, default=DEFAULT_NUTS, help="NUTS boundary GeoPackage")
    p.add_argument(
        "--cntr-code",
        type=str,
        default=None,
        help="Two-letter country code (CNTR_CODE), e.g. EL. If omitted, use full extent of --nuts.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .gpkg path")
    p.add_argument("--layer", type=str, default="osm_energy_polygons", help="GPKG layer name")
    p.add_argument(
        "--min-polygon-area-m2",
        type=float,
        default=10.0,
        help="Drop polygons with area below this in TARGET_CRS after clip (default: 10, from global schema).",
    )
    p.add_argument("--osmium", type=str, default=None, help="Path to osmium-tool executable (optional)")
    p.add_argument("--no-bbox-extract", action="store_true", help="Do not run osmium extract by WGS84 bbox")
    p.add_argument(
        "--prefilter-tags",
        action="store_true",
        help="Run osmium tags-filter before pyosmium (smaller temp PBF; can produce zero features for multipolygons)",
    )
    p.add_argument("--dedupe-geometry", action="store_true", help="Drop duplicate geometries (WKB identical)")
    p.add_argument("--keep-temp", action="store_true", help="Keep intermediate PBFs in a temp directory")
    p.add_argument(
        "--allow-large-pbf-without-osmium",
        action="store_true",
        help="Allow reading a large .pbf without osmium-tool (likely MemoryError or very slow)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print counts and sample tag lines through the area callback and post-processing",
    )
    p.add_argument(
        "--debug-samples",
        type=int,
        default=5,
        metavar="N",
        help="With --debug, max sample lines per bucket (default 5)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pbf = args.pbf.expanduser().resolve()
    if not pbf.is_file():
        raise SystemExit(f"PBF not found: {pbf}")

    nuts = args.nuts.expanduser().resolve()
    if not nuts.is_file():
        raise SystemExit(f"Boundary file not found: {nuts}")

    boundary, nuts_n = _load_boundary(nuts, args.cntr_code)
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
            f"Input PBF is large ({pbf.stat().st_size / (1024**3):.1f} GiB) and osmium-tool was not found.\n"
            "Parsing all of Europe (or similar) in one pass loads a massive node index in memory and "
            "typically fails with MemoryError.\n"
            "Install osmium-tool (e.g. conda install -c conda-forge osmium-tool) so this script can run "
            "osmium extract (bbox) and osmium tags-filter first, or point --pbf to a small country extract.\n"
            "If you really want to try the full file: pass --allow-large-pbf-without-osmium"
        )

    try:
        if osmium_exe and not args.no_bbox_extract:
            temp_root = Path(tempfile.mkdtemp(prefix="osm_energy_"))
            bbox_pbf = temp_root / "bbox_extract.osm.pbf"
            _extract_bbox(osmium_exe, bbox_wgs84, pbf, bbox_pbf)
            work_pbf = bbox_pbf

        if osmium_exe and args.prefilter_tags:
            if temp_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix="osm_energy_"))
            filtered = temp_root / "tags_filtered.osm.pbf"
            _tags_filter(osmium_exe, work_pbf, filtered, OSMIUM_TAG_FILTERS)
            work_pbf = filtered
        else:
            if not osmium_exe:
                warnings.warn(
                    "osmium-tool not found: reading the full PBF without bbox/tags prefilter can be very slow. "
                    "Install conda-forge::osmium-tool or pass --osmium PATH.",
                    stacklevel=1,
                )
            elif args.no_bbox_extract:
                warnings.warn(
                    "Skipping bbox extract: parsing may be slow or memory-heavy.",
                    stacklevel=1,
                )

        parse_debug = AreaParseDebug() if args.debug else None
        if args.debug:
            _print_debug_header(
                work_pbf=work_pbf,
                bbox_wgs84=bbox_wgs84,
                boundary=boundary,
                nuts_n=nuts_n,
                cntr_code=args.cntr_code,
                prefilter_tags=args.prefilter_tags,
                no_bbox_extract=args.no_bbox_extract,
            )
        collector = EnergyAreaCollector(
            debug=parse_debug,
            debug_sample_limit=args.debug_samples,
        )
        collector.apply_file(str(work_pbf), locations=True, idx="flex_mem")

        if args.debug and parse_debug is not None:
            _print_area_debug(parse_debug)

        if not collector.rows:
            raise SystemExit(
                "No matching polygon areas found (empty result).\n"
                "If you used --prefilter-tags, retry without it: tags-filter often strips multipolygon "
                "geometry when landuse/power tags are only on outer ways.\n"
                "Otherwise verify --pbf, --cntr-code, and that the extract contains tagged polygon features."
            )

        gdf = gpd.GeoDataFrame(collector.rows, geometry="geometry", crs="EPSG:4326")
        if args.debug:
            print(f"[debug] GeoDataFrame from collector: {len(gdf)} rows", flush=True)

        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]

        key = gdf["osm_element_type"].astype(str) + ":" + gdf["osm_element_id"].astype(str)
        gdf = gdf.loc[~key.duplicated(keep="first")].copy()
        if args.debug:
            print(f"[debug] after dedupe (osm_element_type:id): {len(gdf)} rows", flush=True)

        boundary_wgs = boundary.to_crs(4326)
        n_before_clip = len(gdf)
        gdf = gpd.clip(gdf, boundary_wgs)
        if args.debug:
            print(
                f"[debug] after clip to NUTS boundary: {len(gdf)} rows (before: {n_before_clip})",
                flush=True,
            )
        if gdf.empty:
            raise SystemExit("No features left after clipping to boundary.")

        gdf = gdf.to_crs(TARGET_CRS)
        n_before = len(gdf)
        gdf = filter_polygon_min_area(gdf, float(args.min_polygon_area_m2))
        if args.debug:
            print(
                f"[debug] after min polygon area (>={args.min_polygon_area_m2} m2 in {TARGET_CRS}): "
                f"{len(gdf)} rows (dropped {n_before - len(gdf)})",
                flush=True,
            )

        if args.dedupe_geometry:
            gdf["_wkb"] = gdf.geometry.apply(lambda geom: geom.wkb)
            gdf = gdf.drop_duplicates(subset=["_wkb"], keep="first").drop(columns=["_wkb"])

        out = args.out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass

        gdf.to_file(out, layer=args.layer, driver="GPKG")
        print(f"Wrote {len(gdf)} features -> {out} (layer={args.layer!r}, crs={TARGET_CRS})")
    finally:
        if temp_root is not None:
            if args.keep_temp:
                print(f"Kept temp dir: {temp_root}", flush=True)
            else:
                shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
