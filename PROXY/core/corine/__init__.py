"""CORINE legend, raster-window, and class-mask helpers.

Use `encoding` for pixel-code semantics and `raster` for GeoTIFF access on the
shared reference grid. This package groups all CORINE-specific logic that used
to live as top-level `core.corine_pixels` and `core.corine_masks` modules.
"""

from __future__ import annotations

from .encoding import (
    build_clc_indicators,
    decode_corine_to_l3_pixels,
    default_corine_index_map_path,
    load_ordered_l3_codes,
    normalized_corine_pixel_encoding,
    resolve_corine_l3_lut,
)
from .raster import (
    clc_group_masks,
    corine_binary_mask,
    corine_binary_mask_adapted,
    read_corine_window,
    resolve_corine_tif,
    warp_corine_codes_nearest,
)

__all__ = [
    "build_clc_indicators",
    "clc_group_masks",
    "corine_binary_mask",
    "corine_binary_mask_adapted",
    "decode_corine_to_l3_pixels",
    "default_corine_index_map_path",
    "load_ordered_l3_codes",
    "normalized_corine_pixel_encoding",
    "read_corine_window",
    "resolve_corine_l3_lut",
    "resolve_corine_tif",
    "warp_corine_codes_nearest",
]
