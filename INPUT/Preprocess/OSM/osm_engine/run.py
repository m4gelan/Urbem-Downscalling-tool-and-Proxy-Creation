"""Run a sector: ``mode: rules`` (YAML) or ``plugin:`` (industry_v1 / fugitive_v1)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml

from . import common
from .rules_handler import RulesCollector


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "osm_schema.yaml"


def load_schema() -> dict[str, Any]:
    p = _schema_path()
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _offroad_frozensets(sec: dict[str, Any]) -> dict[str, frozenset[str]]:
    raw = sec.get("offroad_sets") or {}
    return {k: frozenset(str(x) for x in v) for k, v in raw.items()}


def run_rules_sector(
    sector_id: str,
    *,
    pbf: Path,
    nuts: Path,
    out: Path,
    cntr_code: str | None,
    osmium_exe: str | None,
    no_bbox_extract: bool,
    allow_large_pbf: bool,
    bbox_extract_pbf: Path | None,
    with_optional: bool = False,
    include_wastewater_plant: bool = True,
) -> None:
    data = load_schema()
    glob = data.get("global") or {}
    sectors = data.get("sectors") or {}
    if sector_id not in sectors:
        raise SystemExit(f"Unknown sector in osm_schema.yaml: {sector_id!r}")
    sec = dict(sectors[sector_id])
    sec["id"] = sector_id
    if sec.get("mode") != "rules":
        raise SystemExit(f"Sector {sector_id} is not mode: rules")

    min_m2 = float(glob.get("min_polygon_area_m2", 10.0))

    pbf = pbf.expanduser().resolve()
    nuts = nuts.expanduser().resolve()
    out = out.expanduser().resolve()
    if not pbf.is_file():
        raise SystemExit(f"PBF not found: {pbf}")
    if not nuts.is_file():
        raise SystemExit(f"NUTS not found: {nuts}")

    boundary, _n = common.load_boundary(nuts, cntr_code)
    bbox_wgs84 = common.bbox_str_wgs84(boundary)
    osmium_exe = common.resolve_osmium_exe(osmium_exe)
    work_pbf = pbf
    temp_root: Path | None = None

    if (
        not osmium_exe
        and pbf.stat().st_size > common.MAX_PBF_BYTES_WITHOUT_OSMIUM_TOOL
        and not allow_large_pbf
    ):
        raise SystemExit(
            f"PBF is large and osmium-tool was not found. Install osmium-tool or pass "
            f"--allow-large-pbf-without-osmium."
        )

    try:
        if osmium_exe and not no_bbox_extract:
            temp_root = Path(tempfile.mkdtemp(prefix=f"osm_{sector_id}_"))
            be = bbox_extract_pbf or (temp_root / "bbox_extract.osm.pbf")
            be = be.expanduser().resolve()
            be.parent.mkdir(parents=True, exist_ok=True)
            common.extract_bbox(osmium_exe, bbox_wgs84, pbf, be)
            work_pbf = be
        elif not osmium_exe:
            warnings.warn(
                "osmium-tool not found: full PBF read is slow / high RAM.",
                stacklevel=1,
            )

        ctx = {
            "with_optional": with_optional,
            "include_wastewater_plant": include_wastewater_plant,
        }
        off_sets = _offroad_frozensets(sec) if sec.get("augment_family") == "offroad" else None
        col = RulesCollector(sec, ctx=ctx, offroad_sets=off_sets)
        col.apply_file(str(work_pbf), locations=True, idx="flex_mem")

        boundary_wgs = boundary.to_crs(4326)
        layer_order = sec.get("layer_order") or sorted(col.buckets.keys())

        out_layers: list[tuple[str, gpd.GeoDataFrame]] = []
        for name in layer_order:
            rows = col.buckets.get(name) or []
            if not rows:
                out_layers.append((name, common.empty_gdf_wgs84()))
                continue
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
            gdf = common.dedupe_osm_id(gdf)
            gt = set(gdf.geometry.geom_type.unique())
            if gt <= {"Point"}:
                gdf = common.clip_points_to_3035(gdf, boundary_wgs)
            elif gt <= {"LineString", "MultiLineString"}:
                gdf = common.clip_lines_to_3035(gdf, boundary_wgs)
            else:
                gdf = common.clip_mixed_to_3035(gdf, boundary_wgs)
            gdf = common.filter_polygon_min_area(gdf, min_m2)
            out_layers.append((name, gdf))

        common.write_gpkg(out, out_layers)
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description="OSM rules engine (YAML mode: rules only).")
    p.add_argument(
        "--sector",
        required=True,
        help="Sector id (waste, solvents, offroad, shipping, aviation)",
    )
    p.add_argument("--pbf", type=Path, required=True)
    p.add_argument("--nuts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cntr-code", type=str, default=None)
    p.add_argument("--osmium", type=str, default=None)
    p.add_argument("--no-bbox-extract", action="store_true")
    p.add_argument("--allow-large-pbf-without-osmium", action="store_true")
    p.add_argument("--bbox-extract-pbf", type=Path, default=None)
    p.add_argument("--with-optional", action="store_true")
    p.add_argument("--no-wastewater-plant", action="store_true")
    args = p.parse_args()

    run_rules_sector(
        args.sector,
        pbf=args.pbf,
        nuts=args.nuts,
        out=args.out,
        cntr_code=args.cntr_code,
        osmium_exe=args.osmium,
        no_bbox_extract=bool(args.no_bbox_extract),
        allow_large_pbf=bool(args.allow_large_pbf_without_osmium),
        bbox_extract_pbf=args.bbox_extract_pbf,
        with_optional=bool(args.with_optional),
        include_wastewater_plant=not bool(args.no_wastewater_plant),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
