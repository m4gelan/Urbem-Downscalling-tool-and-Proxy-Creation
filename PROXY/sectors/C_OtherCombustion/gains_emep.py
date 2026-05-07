"""GAINS / EMEP helpers for **M** (re-exports from ``m_builder`` for stable imports)."""

from __future__ import annotations

from .m_builder.assemble import build_M_for_country, build_m_matrix
from .m_builder.emep_ef import ef_kg_per_tj, load_emep
from .m_builder.gains_activity import index_gains_files, map_gains_row_to_class
from .m_builder.mapping_io import load_gains_mapping

__all__ = [
    "build_m_matrix",
    "build_M_for_country",
    "ef_kg_per_tj",
    "load_emep",
    "index_gains_files",
    "map_gains_row_to_class",
    "load_gains_mapping",
]
