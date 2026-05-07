from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root
from PROXY.sectors.K_Agriculture.pipeline import (
    merge_k_agriculture_pipeline_cfg,
    run_k_agriculture_pipeline,
)

logger = logging.getLogger(__name__)


def build(
    *, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str
) -> dict[str, Any]:
    root = project_root()
    if "_project_root" in sector_cfg:
        root = Path(sector_cfg["_project_root"])
    logger.info("K_Agriculture build: country=%s root=%s", country, root)
    cfg = merge_k_agriculture_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
    )
    out = run_k_agriculture_pipeline(root, cfg)
    out_path = Path(out["output_path"])
    if not out_path.is_file():
        raise RuntimeError(f"K_Agriculture build did not write expected output: {out_path}")
    logger.info("K_Agriculture build complete: %s", out_path)
    return out
