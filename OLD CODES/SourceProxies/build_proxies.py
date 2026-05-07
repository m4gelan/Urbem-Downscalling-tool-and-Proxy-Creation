#!/usr/bin/env python3
"""
Build country-wide sector proxy GeoTIFFs (CORINE reference grid).

  python -m SourceProxies.build_proxies
  python -m SourceProxies.build_proxies --config SourceProxies/config/sectors.json --only waste_j shipping_area
  python -m SourceProxies.build_proxies --validate

Requires: xarray, rasterio, geopandas, shapely, pandas, numpy, netCDF4.
Optional: ``pip install tqdm`` for progress bars (``--no-progress`` to disable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="Build SourceProxies sector GeoTIFFs.")
    ap.add_argument(
        "--config",
        type=Path,
        default=root / "SourceProxies" / "config" / "sectors.json",
        help="JSON config (paths, country, sector list)",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        metavar="SECTOR_ID",
        help="Run only these sector ids (e.g. public_power_area agriculture_area)",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="After area builds, print CAMS-cell sum checks (stderr)",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and stage messages",
    )
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["show_progress"] = not bool(args.no_progress)

    from SourceProxies.builders.agriculture_area import build_agriculture_sourcearea
    from SourceProxies.builders.fugitive_area import build_fugitive_sourcearea
    from SourceProxies.builders.industry_area import build_industry_sourcearea
    from SourceProxies.builders.public_power_area import build_public_power_sourcearea
    from SourceProxies.builders.public_power_point import build_public_power_sourcepoint
    from SourceProxies.builders.solvents_eprtr_point_link import (
        build_solvents_eprtr_point_link,
    )
    from SourceProxies.builders.offroad_area import build_offroad_sourcearea
    from SourceProxies.builders.shipping_area import build_shipping_sourcearea
    from SourceProxies.builders.waste_area import build_waste_sourcearea
    from SourceProxies.grid import first_existing_corine, reference_window_profile, resolve_path
    from SourceProxies.progress_util import note

    corine_path = first_existing_corine(root, cfg.get("paths", {}).get("corine"))
    nuts_gpkg = resolve_path(root, cfg["paths"]["nuts_gpkg"])
    pad_m = float((cfg.get("corine") or {}).get("pad_m", 5000.0))

    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_cntr=str(cfg["country"]["nuts_cntr"]),
        pad_m=pad_m,
    )

    only = frozenset(str(x) for x in args.only) if args.only else None
    builders_map = {
        "public_power_area": build_public_power_sourcearea,
        "agriculture_area": build_agriculture_sourcearea,
        "fugitive_area": build_fugitive_sourcearea,
        "industry_area": build_industry_sourcearea,
        "public_power_point": build_public_power_sourcepoint,
        "solvents_eprtr_point_link": build_solvents_eprtr_point_link,
        "waste_area": build_waste_sourcearea,
        "shipping_area": build_shipping_sourcearea,
        "offroad_area": build_offroad_sourcearea,
    }

    sectors_run = [
        e
        for e in cfg.get("sectors", [])
        if only is None or str(e.get("id", "")) in only
    ]
    try:
        from tqdm import tqdm as _tqdm_sectors
    except ImportError:
        _tqdm_sectors = None

    sector_iter = sectors_run
    if cfg.get("show_progress") and _tqdm_sectors is not None and len(sectors_run) > 1:
        sector_iter = _tqdm_sectors(
            sectors_run,
            desc="Sectors",
            unit="sector",
            file=sys.stderr,
        )

    for entry in sector_iter:
        sid = str(entry.get("id", ""))
        bname = str(entry.get("builder", ""))
        fn = builders_map.get(bname)
        if fn is None:
            raise SystemExit(f"Unknown builder {bname!r} for sector id={sid!r}")

        validate_flag = bool(args.validate) and bname in (
            "public_power_area",
            "agriculture_area",
            "fugitive_area",
            "industry_area",
            "shipping_area",
            "offroad_area",
        )
        if cfg.get("show_progress"):
            note(f"--- {sid} ({bname}) ---")
        if bname in (
            "public_power_area",
            "agriculture_area",
            "fugitive_area",
            "industry_area",
            "shipping_area",
            "offroad_area",
        ):
            out = fn(
                root,
                cfg,
                ref,
                sector_entry=entry,
                run_validate=validate_flag,
            )
        else:
            out = fn(root, cfg, ref, sector_entry=entry)
        print(out)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
