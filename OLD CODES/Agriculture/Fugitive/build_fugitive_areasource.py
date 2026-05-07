#!/usr/bin/env python3
"""CLI: build GNFR D fugitive area-source weight GeoTIFF (multi-band, CAMS-cell normalized)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="Build fugitive area-source weight raster.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to fugitive_area.yaml (default: Fugitive/config/fugitive_area.yaml)",
    )
    ap.add_argument("--quiet", action="store_true", help="Less logging.")
    args = ap.parse_args()

    from Waste.j_waste_weights.io_utils import load_ref_profile

    from Fugitive.config_utils import load_fugitive_yaml
    from Fugitive.pipeline import run_fugitive_pipeline

    cfg_path = args.config
    if cfg_path:
        cfg_p = cfg_path if cfg_path.is_absolute() else (root / cfg_path)
        fugitive_cfg = load_fugitive_yaml(cfg_p.resolve())
    else:
        fugitive_cfg = load_fugitive_yaml(None)
    log_cfg = fugitive_cfg.get("logging") or {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    ref = load_ref_profile(fugitive_cfg)
    fb = str((fugitive_cfg.get("defaults") or {}).get("fallback_country_iso3", "GRC"))
    out = run_fugitive_pipeline(
        root,
        fugitive_cfg,
        ref,
        country_iso3_fallback=fb,
        show_progress=not args.quiet,
    )
    print(out)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
