"""
Build CAMS-scale raw proxies on the CORINE 100 m grid (EPSG:3035 typical).

All inputs and output folder are declared in a JSON file (see proxy_factory.paths.example.json).
No dependency on urbem_v2.

Example:
  python -m urbem_interface.proxies.factory.build_proxies --config urbem_interface/config/proxy_factory.json
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from urbem_interface.proxies.factory.config_loader import load_job
from urbem_interface.proxies.factory.corine_build import build_corine_proxies_on_grid
from urbem_interface.proxies.factory.eprtr_build import (
    build_eprtr_proxies_on_grid,
    normalize_waste_tif_name,
)
from urbem_interface.proxies.factory.reference import load_grid_from_corine_raster
from urbem_interface.proxies.factory.regions import (
    VECTOR_SUBSET_BBOX_3035,
    intersect_projected_bboxes,
)
from urbem_interface.proxies.factory.shipping_merge import build_shipping_proxy_merged_rasters

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("proxy_factory")

_publish_progress_callback = None


def set_publish_progress_callback(cb):
    """Optional (internal_name, final_name) -> None; cleared after build."""
    global _publish_progress_callback
    _publish_progress_callback = cb


def _notify_published(internal_name: str, final_name: str) -> None:
    cb = _publish_progress_callback
    if cb is None:
        return
    try:
        cb(internal_name, final_name)
    except Exception:
        logger.debug("publish progress callback failed", exc_info=True)


def _publish_lu_waste_pair(internal_waste_tif: str) -> tuple[str, str]:
    """Map internal ``lu_waste*.tif`` to published proxy name (pipeline convention)."""
    internal_waste_tif = normalize_waste_tif_name(internal_waste_tif)
    if internal_waste_tif == "lu_waste.tif":
        return ("lu_waste.tif", "Proxy_Waste_Wastewater.tif")
    stem = Path(internal_waste_tif).stem
    if stem.startswith("lu_waste_"):
        return (internal_waste_tif, f"Proxy_Waste_Wastewater_{stem[9:]}.tif")
    return (internal_waste_tif, f"Proxy_{stem}.tif")


def _copy_population(
    src: Path,
    out_dir: Path,
    pop_basename: str,
    *,
    skip_existing: bool = False,
) -> Path | None:
    src = Path(src)
    if not src.exists():
        logger.warning("Population raster missing; skip copy (%s)", src)
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / pop_basename
    if skip_existing and dst.is_file():
        logger.info("Population skipped (--skip-existing): %s exists", dst.name)
        return dst
    shutil.copy2(src, dst)
    logger.info("Copied population -> %s", dst.name)
    return dst


def main(argv: list[str] | None = None) -> int:
    # CORINE GeoTIFFs often embed SRS that disagree slightly with EPSG:3035 registry; prefer EPSG.
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
    os.environ.setdefault("OGR_ORGANIZE_POLYGONS_METHOD", "SKIP")

    p = argparse.ArgumentParser(description="Build raw proxies from JSON config (CORINE grid).")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON file with path_base, paths.*, output_dir, optional crs and bundled JSON overrides.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output directory from config output_dir.",
    )
    p.add_argument("--skip-eprtr", action="store_true")
    eprtr_waste_grp = p.add_mutually_exclusive_group()
    eprtr_waste_grp.add_argument(
        "--eprtr-corine-fallback-only-waste",
        action="store_true",
        help=(
            "Write lu_waste_corine_fallback_only.tif from corine_utm.tif only (CLC classes "
            "matching corine_fallback.snap9); no CORINE FileGDB read, no E-PRTR GeoPackage. "
            "Requires corine_utm.tif + lu_industry.tif. Does not build lu_waste.tif / snap34 / snap1."
        ),
    )
    eprtr_waste_grp.add_argument(
        "--eprtr-facilities-only-waste",
        action="store_true",
        help=(
            "Write lu_waste_eprtr_only.tif: E-PRTR Snap9 point-to-CORINE polygon join only "
            "(no corine_fallback). Full CORINE GDB + E-PRTR GPKG run; does not overwrite lu_waste.tif."
        ),
    )
    eprtr_waste_grp.add_argument(
        "--eprtr-waste-plus-corine-no-ag",
        action="store_true",
        help=(
            "Write lu_waste_eprtr_plus_corine_no_ag.tif: E-PRTR Snap9 joins plus corine_fallback "
            "for snap34/snap1, but omit agricultural snap9 Code_18 fallback; optional snap9_supplement "
            "in eprtr_snap_assignment JSON. Does not overwrite lu_waste.tif."
        ),
    )
    p.add_argument("--skip-shipping", action="store_true")
    p.add_argument("--skip-population", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a phase when its output files already exist (resume after partial runs).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (E-PRTR / CORINE vector reads).",
    )
    p.add_argument(
        "--vector-subset",
        choices=sorted(VECTOR_SUBSET_BBOX_3035.keys()),
        default=None,
        help=(
            "Clip CORINE GDB + E-PRTR vector work to this EPSG:3035 region intersected "
            "with the template grid (faster). Output rasters stay full extent; only cells "
            "in the subset receive polygon burns. Overrides config vector_subset if set."
        ),
    )
    p.add_argument(
        "--waste-tif",
        type=str,
        default=None,
        help=(
            "Basename for the Snap9 / lu_waste raster (default lu_waste.tif). "
            "Example: lu_waste_france.tif. Overrides config waste_output_tif."
        ),
    )
    p.add_argument(
        "--eprtr-snap9-only",
        action="store_true",
        help=(
            "After the E-PRTR vector pipeline, rasterize only the waste layer (Snap9); "
            "skip lu_snap34.tif and lu_snap1.tif. Combine with --vector-subset for faster runs."
        ),
    )
    args = p.parse_args(argv)

    job = load_job(args.config.resolve())
    out_dir = Path(args.out).resolve() if args.out else job.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corine_raster = job.paths["corine_raster"]
    grid = load_grid_from_corine_raster(corine_raster)

    vector_subset = (args.vector_subset or job.vector_subset or "").strip().lower() or None
    if vector_subset is not None and vector_subset not in VECTOR_SUBSET_BBOX_3035:
        logger.error(
            "Unknown vector_subset %r (config or CLI); allowed: %s",
            vector_subset,
            sorted(VECTOR_SUBSET_BBOX_3035.keys()),
        )
        return 1

    eprtr_bbox = grid.projected_bbox
    if vector_subset is not None:
        eprtr_bbox = intersect_projected_bboxes(
            grid.projected_bbox, VECTOR_SUBSET_BBOX_3035[vector_subset]
        )
        logger.info(
            "E-PRTR/CORINE vector domain: subset=%r intersect grid -> xmin=%.0f ymin=%.0f xmax=%.0f ymax=%.0f",
            vector_subset,
            eprtr_bbox["xmin"],
            eprtr_bbox["ymin"],
            eprtr_bbox["xmax"],
            eprtr_bbox["ymax"],
        )

    waste_tif_arg = (args.waste_tif or "").strip() or None
    waste_output_tif = normalize_waste_tif_name(waste_tif_arg or job.waste_output_tif)
    eprtr_snap9_only = bool(args.eprtr_snap9_only or job.eprtr_snap9_only)
    if waste_output_tif != "lu_waste.tif" or eprtr_snap9_only:
        logger.info(
            "E-PRTR waste outputs: file=%s snap9_only=%s",
            waste_output_tif,
            eprtr_snap9_only,
        )

    build_corine_proxies_on_grid(
        corine_raster,
        out_dir,
        grid,
        job.corine_classes,
        skip_existing=args.skip_existing,
    )

    if not args.skip_eprtr:
        build_eprtr_proxies_on_grid(
            job_paths=job.paths,
            bbox=eprtr_bbox,
            crs_projected=job.crs_projected,
            crs_wgs=job.crs_wgs,
            crs_corine=job.crs_corine,
            eprtr_snap_assignment=job.eprtr_snap_assignment,
            keep_intermediate_shp=job.keep_intermediate_shp,
            proxies_folder=out_dir,
            skip_existing=args.skip_existing,
            show_progress=not args.no_progress,
            corine_fallback_only_waste=args.eprtr_corine_fallback_only_waste,
            eprtr_facilities_only_waste=args.eprtr_facilities_only_waste,
            waste_eprtr_plus_corine_no_ag=args.eprtr_waste_plus_corine_no_ag,
            waste_output_tif=waste_output_tif,
            eprtr_snap9_only=eprtr_snap9_only,
        )

    internal_shipping = out_dir / "lu_shipping_with ports.tif"
    if not args.skip_shipping:
        build_shipping_proxy_merged_rasters(
            grid.projected_bbox,
            job.crs_projected,
            job.crs_wgs,
            job.paths["shipping_routes_shp"],
            out_dir,
            out_dir / "lu_industry.tif",
            internal_shipping,
            skip_existing=args.skip_existing,
        )

    if not args.skip_population:
        _copy_population(
            job.paths["population_raster"],
            out_dir,
            job.population_output_basename,
            skip_existing=args.skip_existing,
        )

    waste_internal, waste_published = _publish_lu_waste_pair(waste_output_tif)
    final_map = [
        ("lu_industry.tif", "Proxy_Industry.tif"),
        ("lu_agriculture.tif", "Proxy_Agriculture.tif"),
        ("lu_airport.tif", "Proxy_Aviation.tif"),
        ("lu_offroad.tif", "Proxy_OffRoad_Mobility.tif"),
        (waste_internal, waste_published),
        (internal_shipping.name, "Proxy_Shipping.tif"),
        ("lu_snap34.tif", "Proxy_EPRTR_SNAP34.tif"),
        ("lu_snap1.tif", "Proxy_EPRTR_SNAP1.tif"),
    ]

    for internal_name, final_name in final_map:
        src = out_dir / internal_name
        if not src.exists():
            logger.debug("Skip final %s (missing %s)", final_name, internal_name)
            continue
        dst = out_dir / final_name
        if args.skip_existing and dst.is_file():
            logger.debug("Publish skipped (--skip-existing): %s", final_name)
            continue
        shutil.copy2(src, dst)
        logger.info("Published %s", final_name)
        _notify_published(internal_name, final_name)

    logger.info("Done. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
