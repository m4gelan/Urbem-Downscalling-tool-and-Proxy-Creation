"""Compatibility entrypoint for the K_Agriculture build pipeline.

New code should prefer ``PROXY.sectors.K_Agriculture.pipeline``. This module keeps the
historical ``run_k_agriculture_build`` symbol for external callers during cleanup.
"""
from __future__ import annotations

from .pipeline import merge_k_agriculture_pipeline_cfg, run_k_agriculture_pipeline


def run_k_agriculture_build(root, path_cfg, sector_cfg, *, country):
    cfg = merge_k_agriculture_pipeline_cfg(root, path_cfg, sector_cfg, country=country)
    return run_k_agriculture_pipeline(root, cfg)

__all__ = ["run_k_agriculture_build"]
