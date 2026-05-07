#!/usr/bin/env python3
"""
Split an existing OSM ``.osm.pbf`` into roads and landuse/buildings extracts (no download).

Default input is ``<out-dir>/_source/europe-latest.osm.pbf``. Use ``--greece`` for a Greece
preset (expects ``greece-latest.osm.pbf`` unless ``--from-europe`` is used).

With ``--greece --from-europe EUROPE.pbf``, runs ``osmium extract`` on a WGS84 bbox covering
Greece (one sequential read of the Europe file), then ``tags-filter`` on the small extract.

Requires **osmium-tool** on PATH.

On Windows, conda’s ``osmium.exe`` is spawned with ``executable=`` and argv[0] ``osmium``.

Usage (from project root):
  python Solvents/Auxiliaries/download_osm_europe.py
  python Solvents/Auxiliaries/download_osm_europe.py --source D:\\osm\\europe-latest.osm.pbf
  python Solvents/Auxiliaries/download_osm_europe.py --greece
  python Solvents/Auxiliaries/download_osm_europe.py --greece --from-europe data/OSM/_source/europe-latest.osm.pbf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "OSM"

# WGS84: west,south,east,north (osmium extract -b). Covers mainland, islands, Crete, east Aegean.
GREECE_BBOX = "19.05,34.40,30.25,41.95"


def _human_mb(n: float) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def _resolve_osmium_exe(cli_path: str | None) -> str:
    if cli_path:
        p = Path(cli_path).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--osmium does not exist: {p}")
        return str(p)
    exe = shutil.which("osmium")
    if not exe:
        raise SystemExit(
            "osmium was not found on PATH. Install osmium-tool and retry.\n"
            "Example: conda install -c conda-forge osmium-tool"
        )
    return str(Path(exe).resolve())


def _run_osmium(osmium_exe: str, *tail: str) -> None:
    if sys.platform == "win32":
        cmd = ["osmium", *tail]
        print("Running:", " ".join(cmd), f"(executable={osmium_exe})", flush=True)
        r = subprocess.run(cmd, executable=osmium_exe, check=False)
    else:
        cmd = [osmium_exe, *tail]
        print("Running:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"osmium failed with exit code {r.returncode}: {' '.join(tail[:3])} ...")


def _run_osmium_tags_filter(osmium_exe: str, source: Path, out: Path, *filter_args: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_osmium(osmium_exe, "tags-filter", "--overwrite", "-o", str(out), str(source), *filter_args)
    print(f"Wrote {_human_mb(out.stat().st_size)} -> {out}", flush=True)


def _run_osmium_extract_bbox(osmium_exe: str, bbox: str, source: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_osmium(osmium_exe, "extract", "--overwrite", "-b", bbox, "-o", str(out), str(source))
    print(f"Wrote {_human_mb(out.stat().st_size)} -> {out}", flush=True)


def main() -> None:
    default_europe = DEFAULT_OUT_DIR / "_source" / "europe-latest.osm.pbf"
    default_greece = DEFAULT_OUT_DIR / "_source" / "greece-latest.osm.pbf"

    p = argparse.ArgumentParser(
        description="Split OSM PBF into roads / landuse+buildings extracts (optional Greece bbox from Europe)."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for outputs (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Input .osm.pbf (default: Europe {default_europe}, or Greece {default_greece} with --greece)",
    )
    p.add_argument(
        "--greece",
        action="store_true",
        help="Greece preset: output OSM_*_Greece.osm.pbf; default source is greece-latest.osm.pbf.",
    )
    p.add_argument(
        "--from-europe",
        type=Path,
        default=None,
        metavar="PBF",
        help="With --greece: clip Greece from this Europe .osm.pbf (one full read), then filter.",
    )
    p.add_argument(
        "--keep-greece-extract",
        action="store_true",
        help="With --from-europe: keep the intermediate bbox-clipped PBF under _work/.",
    )
    p.add_argument(
        "--osmium",
        type=str,
        default=None,
        metavar="EXE",
        help="Path to osmium.exe (default: resolve 'osmium' on PATH).",
    )
    p.add_argument(
        "--delete-source-after-filter",
        action="store_true",
        help="Remove the resolved input PBF after both filters succeed (not the Europe file when using --from-europe).",
    )
    args = p.parse_args()

    if args.from_europe and not args.greece:
        raise SystemExit("--from-europe requires --greece")

    out_dir: Path = args.out_dir.expanduser().resolve()
    osmium_exe = _resolve_osmium_exe(args.osmium)

    intermediate: Path | None = None
    if args.greece and args.from_europe:
        europe = args.from_europe.expanduser().resolve()
        if not europe.is_file():
            raise SystemExit(f"--from-europe file not found: {europe}")
        work_dir = out_dir / "_work"
        intermediate = work_dir / "greece_bbox_from_europe.osm.pbf"
        print(
            "Clipping Greece bbox from Europe (one full read of the Europe extract; can take many hours).",
            flush=True,
        )
        _run_osmium_extract_bbox(osmium_exe, GREECE_BBOX, europe, intermediate)
        source = intermediate
    else:
        if args.source is not None:
            source = args.source.expanduser().resolve()
        elif args.greece:
            source = default_greece
        else:
            source = default_europe

        if not source.is_file():
            hint = (
                f"For Greece-only from Geofabrik, place greece-latest.osm.pbf at:\n  {default_greece}\n"
                if args.greece
                else f"Place europe-latest.osm.pbf at:\n  {default_europe}\n"
            )
            raise SystemExit(f"Source PBF not found: {source}\n{hint}")

    if args.greece:
        roads_out = out_dir / "OSM_roads_Greece.osm.pbf"
        landuse_out = out_dir / "OSM_landuse_buildings_Greece.osm.pbf"
    else:
        roads_out = out_dir / "OSM_roads.osm.pbf"
        landuse_out = out_dir / "OSM_landuse_buildings.osm.pbf"

    _run_osmium_tags_filter(osmium_exe, source, roads_out, "w/highway")
    _run_osmium_tags_filter(
        osmium_exe,
        source,
        landuse_out,
        "w/landuse",
        "w/building",
        "r/landuse",
        "r/building",
    )

    if intermediate and intermediate.is_file() and not args.keep_greece_extract:
        print(f"Removing intermediate extract: {intermediate}", flush=True)
        intermediate.unlink()
        try:
            intermediate.parent.rmdir()
        except OSError:
            pass

    if args.delete_source_after_filter and source.is_file():
        print(f"Removing source file: {source}", flush=True)
        source.unlink()

    print("Done.", flush=True)
    print(f"  Roads:           {roads_out}", flush=True)
    print(f"  Landuse/build:   {landuse_out}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
