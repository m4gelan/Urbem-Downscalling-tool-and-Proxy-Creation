"""NFR codes for GNFR C (Other Stationary Combustion / 1A4 family)."""

from __future__ import annotations

NFR_STATIONARY: tuple[str, ...] = ("1A4ai", "1A4bi", "1A4ci")
NFR_OFFROAD: tuple[str, ...] = ("1A4aii", "1A4bii", "1A4cii")
NFR_ALL_C: tuple[str, ...] = NFR_STATIONARY + NFR_OFFROAD
