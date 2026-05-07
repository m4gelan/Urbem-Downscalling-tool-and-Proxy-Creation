"""
Per-process source relevance: each module exposes compute_rho_df(extent_df, cfg).

Shared LUCAS + CORINE point prep is in lucas_points.py. Pathway logic (grazing, manure, synthetic N)
lives in enteric_grazing.py, manure.py, fertilized_land.py. rho = mu / max mu within country (eq. 4).
"""

from __future__ import annotations

__all__: list[str] = []
