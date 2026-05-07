from __future__ import annotations

from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root

from .pipeline import merge_shipping_pipeline_cfg, run_shipping_area_pipeline


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    GNFR G shipping: one area-source weight layer (EMODnet + CORINE ports + OSM), CAMS-cell normalized.
    All pollutants use the same spatial pattern; there is no per-pollutant alpha split.
    """
    root = project_root()
    out_dir = Path(sector_cfg["output_dir"]).resolve()
    cfg = merge_shipping_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
        output_dir=out_dir,
    )
    return run_shipping_area_pipeline(root, cfg)
