"""Industrial / commercial CORINE (CLC 3 and 121 treated as the same class for this project)."""

from __future__ import annotations

import numpy as np

# EEA standard 121 = industrial or commercial; this raster uses internal code 3 for the same class.
INDUSTRIAL_COMMERCIAL_CODES: frozenset[int] = frozenset({3, 121})

_INDUSTRIAL_RGB: tuple[int, int, int] = (21, 101, 192)


def industrial_commercial_hex() -> str:
    """Single map colour for CLC 3 or 121 (industrial / commercial units)."""
    r, g, b = _INDUSTRIAL_RGB
    return f"#{r:02x}{g:02x}{b:02x}"


def is_industrial_commercial_clc(c: int) -> bool:
    return int(c) in INDUSTRIAL_COMMERCIAL_CODES


def corine_clc_overlay_rgba(
    clc: np.ndarray,
    *,
    highlight_codes: tuple[int, ...] = (3, 121),
    base_alpha_highlight: int = 235,
    rgb: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Raster overlay: only ``highlight_codes`` pixels are drawn; others stay transparent.

    Used for the Public Power map: CORINE GeoTIFF resampled to the Folium grid (nearest).
    Optional ``rgb`` overrides the default industrial blue for multi-layer maps (e.g. Offroad).
    """
    hl = frozenset(int(x) for x in highlight_codes)
    h, w = clc.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = clc >= 0
    if not np.any(valid):
        return rgba

    r, g, b = rgb if rgb is not None else _INDUSTRIAL_RGB
    for c in hl:
        m = valid & (clc == int(c))
        if np.any(m):
            rgba[m, 0] = r
            rgba[m, 1] = g
            rgba[m, 2] = b
            rgba[m, 3] = base_alpha_highlight
    return rgba
