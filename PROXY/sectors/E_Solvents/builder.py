from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root

from PROXY.sectors.E_Solvents.pipeline import merge_solvents_pipeline_cfg, run_solvents_pipeline

logger = logging.getLogger(__name__)


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    GNFR E solvent area weights: in-repo pipeline (archetypes, beta, CEIP, CAMS-E).

    Config under ``sector_cfg["solvents"]`` overlays
    ``PROXY/config/ceip/profiles/solvents_pipeline.yaml``.
    """
    root = project_root()
    output_path = Path(sector_cfg["output_path"]).resolve()

    logger.info(
        "E_Solvents build: country=%s output_path=%s",
        country,
        output_path,
    )

    cfg = merge_solvents_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
        output_path=output_path,
    )

    paths = cfg.get("paths") or {}
    logger.info(
        "E_Solvents build: merged cfg — cams_nc=%s corine=%s population_tif=%s "
        "ceip_workbook=%s osm_solvent_gpkg=%s",
        paths.get("cams_nc"),
        paths.get("corine"),
        paths.get("population_tif"),
        paths.get("ceip_workbook"),
        paths.get("osm_solvent_gpkg"),
    )

    out = run_solvents_pipeline(root, cfg, config_path=None)
    out_tif = Path(out["output_tif"]).resolve()
    if out_tif != output_path:
        raise RuntimeError(
            f"E_Solvents build wrote {out_tif} but sector output_path is {output_path}"
        )
    logger.info("E_Solvents build finished: output_path=%s", output_path)
    return {"output_path": str(output_path)}
