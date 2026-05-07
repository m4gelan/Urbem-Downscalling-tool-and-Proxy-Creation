"""GNFR C **M** matrix: GAINS activity, EMEP factors, Eurostat end-use scaling."""

from __future__ import annotations

from .assemble import build_m_matrix, build_M_for_country
from .emep_ef import ef_kg_per_tj, load_emep
from .enduse_factors import EndUseFactors, activity_share_by_class, compute_end_use_factors, log_enduse_tables
from .gains_activity import index_gains_files, load_gains_rows, map_gains_row_to_class
from .mapping_io import load_gains_mapping

__all__ = [
    "activity_share_by_class",
    "build_m_matrix",
    "build_M_for_country",
    "compute_end_use_factors",
    "EndUseFactors",
    "ef_kg_per_tj",
    "load_emep",
    "index_gains_files",
    "load_gains_mapping",
    "load_gains_rows",
    "log_enduse_tables",
    "map_gains_row_to_class",
]
