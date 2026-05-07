#!/usr/bin/env python3
"""
Extract waste-related OSM features from a landuse/buildings PBF using osmium-tool.

Requires **osmium** on PATH (e.g. ``conda install -c conda-forge osmium-tool``).

Writes a GeoPackage suitable for ``paths.osm_waste_gpkg`` in ``Waste/j_waste_weights/config.yaml``.

Example::

  python Waste/Auxiliaries/osm_waste_landuse_extract.py \\
    --pbf data/OSM/OSM_landuse_buildings_Greece.osm.pbf \\
    --out Waste/outputs/osm_waste_features.gpkg

The filter selects multipolygons tagged with common waste-related landuse/amenity keys.
Adjust ``TAG_FILTERS`` if your OSM tagging conventions differ.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _resolve_osmium() -> str:
    exe = shutil.which("osmium")
    if not exe:
        raise SystemExit(
            "osmium not found on PATH. Install osmium-tool (e.g. conda-forge) and retry."
        )
    return exe


def main() -> int:
    p = argparse.ArgumentParser(description="Extract OSM waste-related polygons to GeoPackage.")
    p.add_argument("--pbf", type=Path, required=True, help="Input OSM PBF (landuse/buildings).")
    p.add_argument("--out", type=Path, required=True, help="Output .gpkg path.")
    p.add_argument(
        "--ogr2ogr",
        type=str,
        default="ogr2ogr",
        help="ogr2ogr executable (used to convert filtered OSM to GPKG).",
    )
    args = p.parse_args()
    if not args.pbf.is_file():
        raise SystemExit(f"PBF not found: {args.pbf}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    osmium = _resolve_osmium()

    tags = [
        "landuse=landfill",
        "landuse=dump",
        "amenity=waste_transfer_station",
        "amenity=recycling",
        "industrial=waste",
    ]

    with tempfile.TemporaryDirectory() as td:
        filtered = Path(td) / "filtered.osm.pbf"
        cmd = [
            osmium,
            "tags-filter",
            "--overwrite",
            "-o",
            str(filtered),
            str(args.pbf),
            *tags,
        ]
        print("Running:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            return r.returncode
        ogr_cmd = [
            args.ogr2ogr,
            "-overwrite",
            "-f",
            "GPKG",
            str(args.out),
            str(filtered),
            "multipolygons",
        ]
        print("Running:", " ".join(ogr_cmd), flush=True)
        r2 = subprocess.run(ogr_cmd, check=False)
        if r2.returncode != 0:
            print(
                "ogr2ogr failed. Install GDAL with GeoPackage support, or convert manually.",
                file=sys.stderr,
            )
            return r2.returncode
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
