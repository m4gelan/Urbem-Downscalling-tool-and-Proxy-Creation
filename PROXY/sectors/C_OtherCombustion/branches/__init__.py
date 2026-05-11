"""GNFR C spatial subgroups: stationary + three off-road proxies."""

from __future__ import annotations

from .commercial_offroad import compute_W_B
from .forestry_offroad import compute_W_F
from .residential_offroad import compute_W_R
from .stationary import compute_W_stat

__all__ = [
    "compute_W_stat",
    "compute_W_F",
    "compute_W_R",
    "compute_W_B",
]
