"""Shared proxy helpers (CORINE × population, etc.)."""

from __future__ import annotations

from .eligibility_pop_blend import (
    pop_01_within_cell,
    share_tensor_eligibility_pop_blend,
)

__all__ = [
    "pop_01_within_cell",
    "share_tensor_eligibility_pop_blend",
]
