"""
UrbEm Interface - CLI entry point.
Use this when running from command line; use launcher.py for the desktop UI.

Usage:
  python urbem_interface/run.py urbem_interface/config/ioannina_2019.json
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_here = Path(__file__).resolve().parent

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append(str(_here / "config" / "ioannina_2019.json"))

    cfg = Path(sys.argv[1])
    if not cfg.exists():
        sys.exit(f"Config not found: {cfg}")

    config_dir = cfg.parent
    proxies_cfg = config_dir / "proxies.json"
    snap_cfg = config_dir / "snap_mapping.json"
    if not proxies_cfg.exists():
        sys.exit(f"Proxies config not found: {proxies_cfg}")
    if not snap_cfg.exists():
        sys.exit(f"SNAP mapping not found: {snap_cfg}")
    points_cfg = config_dir / "pointsources.json"
    lines_cfg = config_dir / "linesources.json"

    from urbem_interface.utils.config_loader import load_run_config
    from urbem_interface.logging_config import get_logger

    logger = get_logger("urbem_interface.run")

    try:
        logger.info("Loading configs...")
        run_cfg = load_run_config(cfg)
        if str(run_cfg.get("proxy_pipeline_report", "")).lower() in ("1", "true", "yes"):
            from urbem_interface.pipeline import (
                build_proxy_pipeline_bundle,
                merge_proxy_catalog,
                render_proxy_pipeline_report,
            )
            from urbem_interface.utils.config_loader import load_proxies_config, resolve_paths

            paths = resolve_paths(run_cfg, config_dir)
            pc = merge_proxy_catalog(
                load_proxies_config(proxies_cfg, proxies_folder=paths["proxies_folder"])
            )
            bundle = build_proxy_pipeline_bundle(pc)
            print(
                render_proxy_pipeline_report(
                    bundle,
                    proxies_folder=paths["proxies_folder"],
                )
            )
        source_type = str(run_cfg.get("source_type", "area")).lower()
        if source_type == "area":
            from urbem_interface.core.area_sources import run_and_export
            out_path = run_and_export(cfg, proxies_cfg, snap_cfg)
        elif source_type == "point":
            if not points_cfg.exists():
                sys.exit(f"Point-sources config not found: {points_cfg}")
            from urbem_interface.core.point_sources import run_and_export
            out_path = run_and_export(cfg, points_cfg, snap_cfg)
        elif source_type == "line":
            if not lines_cfg.exists():
                sys.exit(f"Line-sources config not found: {lines_cfg}")
            from urbem_interface.core.line_sources import run_and_export
            out_path = run_and_export(cfg, lines_cfg, proxies_cfg)
        else:
            sys.exit(f"Unknown source_type: {source_type!r} (expected area|point|line)")
        logger.info(f"Done. Wrote {out_path}")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
