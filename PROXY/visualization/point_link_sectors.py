"""Sector keys with CAMS point ↔ facility match artifacts (CSV + optional 2-band link GeoTIFF).

Used by ``python -m PROXY.main visualize --point-link`` and by
:func:`PROXY.visualization.point_link_context_map.write_point_link_context_html`.
Add a sector here when it gains ``match-points`` support and point-context HTML.
"""

from __future__ import annotations

POINT_LINK_SECTOR_KEYS: frozenset[str] = frozenset(
    ("A_PublicPower", "B_Industry", "D_Fugitive", "E_Solvents", "J_Waste", "H_Aviation")
)
