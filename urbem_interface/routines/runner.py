"""
Pipeline runner - executes UrbEm pipelines with progress callbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def get_pipeline_stages(source_type: str) -> list[dict]:
    """Return list of pipeline stages for the given source type."""
    stages = {
        "area": [
            {"id": "domain", "label": "Domain loaded"},
            {"id": "cams", "label": "CAMS grid reprojected"},
            {"id": "proxies", "label": "Proxies loaded"},
            {"id": "downscale", "label": "Emissions downscaled"},
            {"id": "export", "label": "Export complete"},
        ],
        "point": [
            {"id": "domain", "label": "Domain loaded"},
            {"id": "cams", "label": "CAMS point/area loaded"},
            {"id": "points", "label": "Points assigned"},
            {"id": "distribute", "label": "Emissions distributed"},
            {"id": "export", "label": "Export complete"},
        ],
        "line": [
            {"id": "domain", "label": "Domain loaded"},
            {"id": "cams", "label": "CAMS reprojected"},
            {"id": "proxies", "label": "Population proxy"},
            {"id": "osm", "label": "OSM roads fetched"},
            {"id": "lines", "label": "Emissions to lines"},
            {"id": "export", "label": "Export complete"},
        ],
    }
    return stages.get(source_type, stages["area"])


def run_pipeline(
    run_config_path: Path,
    config_dir: Path,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> Path:
    """
    Run the UrbEm pipeline based on source_type in run config.
    Calls progress_callback(stage_id, data) at each stage.
    Returns output file path.
    """
    from urbem_interface.utils.config_loader import load_run_config

    run_config = load_run_config(run_config_path)
    source_type = str(run_config.get("source_type", "area")).lower()

    def emit(stage_id: str, data: dict | None = None):
        if progress_callback:
            progress_callback(stage_id, data or {})

    emit("domain", {"bounds": run_config.get("domain", {})})

    if source_type == "area":
        from urbem_interface.core.area_sources import run_and_export as run_area

        proxies_cfg = config_dir / "proxies.json"
        snap_cfg = config_dir / "snap_mapping.json"
        if not proxies_cfg.exists():
            raise FileNotFoundError(f"Proxies config not found: {proxies_cfg}")
        if not snap_cfg.exists():
            raise FileNotFoundError(f"SNAP mapping not found: {snap_cfg}")
        out_path = run_area(run_config_path, proxies_cfg, snap_cfg, progress_callback=emit)
        emit("export", {"output_path": str(out_path)})
        return out_path

    if source_type == "point":
        from urbem_interface.core.point_sources import run_and_export as run_point

        points_cfg = config_dir / "pointsources.json"
        snap_cfg = config_dir / "snap_mapping.json"
        if not points_cfg.exists():
            raise FileNotFoundError(f"Point-sources config not found: {points_cfg}")
        if not snap_cfg.exists():
            raise FileNotFoundError(f"SNAP mapping not found: {snap_cfg}")
        out_path = run_point(run_config_path, points_cfg, snap_cfg)
        emit("export", {"output_path": str(out_path)})
        return out_path

    if source_type == "line":
        from urbem_interface.core.line_sources import run_and_export as run_line

        lines_cfg = config_dir / "linesources.json"
        proxies_cfg = config_dir / "proxies.json"
        if not lines_cfg.exists():
            raise FileNotFoundError(f"Line-sources config not found: {lines_cfg}")
        if not proxies_cfg.exists():
            raise FileNotFoundError(f"Proxies config not found: {proxies_cfg}")
        out_path = run_line(run_config_path, lines_cfg, proxies_cfg, progress_callback=emit)
        emit("export", {"output_path": str(out_path)})
        return out_path

    raise ValueError(f"Unknown source_type: {source_type}")
