"""
Build GNFR G fine-grid area weights (legacy import path).

Canonical implementation: :mod:`PROXY.sectors.G_Shipping.proxy_shipping`.
"""

from __future__ import annotations

from PROXY.sectors.G_Shipping.proxy_shipping import (  # noqa: F401
    build_combined_proxy,
    build_ref_corine_nuts,
    corine_port_fraction_and_codes,
    ensure_ref_window_bounds,
    load_ref_from_fine_grid_tif,
    minmax01,
    resolve_corine_tif,
    run_shipping_areasource,
    warp_corine_codes_nearest,
    write_diagnostic_rasters,
)

__all__ = [
    "build_combined_proxy",
    "build_ref_corine_nuts",
    "corine_port_fraction_and_codes",
    "ensure_ref_window_bounds",
    "load_ref_from_fine_grid_tif",
    "minmax01",
    "resolve_corine_tif",
    "run_shipping_areasource",
    "warp_corine_codes_nearest",
    "write_diagnostic_rasters",
]
