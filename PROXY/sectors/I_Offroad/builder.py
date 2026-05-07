"""I_Offroad sector entrypoint.

The former ``multiband_builder.py`` catch-all has been split into this thin builder
and ``pipeline.py``.  The builder stays focused on the public ``PROXY.main`` contract:
merge config, log the resolved output target, run the pipeline, and verify that the
expected GeoTIFF was written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root, resolve_path
from PROXY.sectors.I_Offroad.pipeline import merge_offroad_pipeline_cfg, run_offroad_pipeline

logger = logging.getLogger(__name__)


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """Build GNFR I multiband area weights for the requested NUTS country window."""
    root = project_root()
    output_path = resolve_path(root, Path(sector_cfg["output_path"])).resolve()
    logger.info("I_Offroad build: country=%s output_path=%s", country, output_path)

    cfg = merge_offroad_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
        output_path=output_path,
    )
    paths = cfg.get("paths") or {}
    logger.info(
        "I_Offroad build: merged cfg — cams_nc=%s corine=%s osm=%s ceip=%s",
        paths.get("cams_nc"),
        paths.get("corine"),
        paths.get("osm_gpkg"),
        paths.get("ceip_workbook"),
    )

    out = run_offroad_pipeline(root, cfg)
    out_tif = Path(out["output_path"]).resolve()
    if out_tif != output_path:
        raise RuntimeError(
            f"I_Offroad build wrote {out_tif} but sector output_path is {output_path}"
        )
    logger.info("I_Offroad build complete: %s", output_path)
    return {"output_path": str(output_path), "manifest_path": str(out.get("manifest_path", ""))}
