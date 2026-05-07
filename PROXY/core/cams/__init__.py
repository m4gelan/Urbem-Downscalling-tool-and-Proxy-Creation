"""CAMS grid, cell-id, and source-mask helpers.

This package is the canonical home for logic that operates directly on CAMS
NetCDF geography, source rows, country ids, GNFR indices, and fine-grid CAMS
cell-id rasters.
"""
from __future__ import annotations

from .cell_id import build_cams_cell_id_raster
from .domain import country_index_1based, domain_mask_wgs84
from .gnfr import GNFR_ORDER, gnfr_code_to_index
from .grid import (
    build_cam_cell_id,
    build_cam_cell_id_masked_for_sources,
    build_cams_source_index_grid,
    build_cams_source_index_grid_any_gnfr,
    cams_source_mask,
    cams_source_mask_any_gnfr,
    geographic_cids_for_sources,
    read_cams_bounds,
)
from .mask import cams_gnfr_country_source_mask, other_combustion_area_mask

__all__ = [
    "GNFR_ORDER",
    "build_cam_cell_id",
    "build_cams_cell_id_raster",
    "build_cam_cell_id_masked_for_sources",
    "build_cams_source_index_grid",
    "build_cams_source_index_grid_any_gnfr",
    "cams_gnfr_country_source_mask",
    "cams_source_mask",
    "cams_source_mask_any_gnfr",
    "country_index_1based",
    "domain_mask_wgs84",
    "gnfr_code_to_index",
    "geographic_cids_for_sources",
    "other_combustion_area_mask",
    "read_cams_bounds",
]
