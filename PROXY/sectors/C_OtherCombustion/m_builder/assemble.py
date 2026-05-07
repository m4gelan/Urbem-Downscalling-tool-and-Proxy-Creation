"""
Assemble **M[pollutant, class]** from GAINS rows, EMEP EFs, and :class:`EndUseFactors`.

**Formula** (per GAINS row mapped to class ``k``):

``M[p,k] += share × row_multiplier(k) × EF(p, fuel, appliance)``

where ``row_multiplier`` applies Eurostat bucket × GAINS appliance split, or legacy
per-class scalars when Eurostat is disabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..constants import CLASS_TO_INDEX, MODEL_CLASSES
from .emep_ef import ef_kg_per_tj
from .enduse_factors import EndUseFactors
from .gains_activity import _load_gains_table, _norm_pct_cell, map_gains_row_to_class


def build_m_matrix(
    gains_path: Path | None,
    year_col: str,
    mapping_rules: list[dict[str, Any]],
    factors: EndUseFactors,
    emep: dict[str, Any],
    pollutant_outputs: list[str],
    *,
    emep_fuel_hints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    n_p = len(pollutant_outputs)
    n_k = len(MODEL_CLASSES)
    M = np.zeros((n_p, n_k), dtype=np.float64)
    if gains_path is None or not gains_path.is_file():
        return M
    _, rows = _load_gains_table(gains_path, year_col)
    for fuel, app, ycell in rows:
        cls = map_gains_row_to_class(fuel, app, mapping_rules)
        if cls is None:
            continue
        share = _norm_pct_cell(ycell)
        if share <= 0:
            continue
        mult = factors.row_multiplier(cls)
        a_corr = share * mult
        ki = CLASS_TO_INDEX[cls]
        for pi, pol in enumerate(pollutant_outputs):
            ef = ef_kg_per_tj(pol, fuel, app, emep, emep_fuel_hints=emep_fuel_hints)
            M[pi, ki] += a_corr * ef
    return M


def build_M_for_country(
    iso3: str,
    gains_path: Path | None,
    year_col: str,
    mapping_rules: list[dict[str, Any]],
    f_enduse: dict[str, float],
    emep: dict[str, Any],
    pollutant_outputs: list[str],
    *,
    emep_fuel_hints: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Backward-compatible signature: ``f_enduse`` is per-class legacy scalars."""
    _ = iso3
    factors = EndUseFactors.from_legacy_class_scalars(f_enduse)
    return build_m_matrix(
        gains_path,
        year_col,
        mapping_rules,
        factors,
        emep,
        pollutant_outputs,
        emep_fuel_hints=emep_fuel_hints,
    )
