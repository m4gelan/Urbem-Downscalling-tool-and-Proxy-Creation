"""Visualization helper for legacy/offroad subsector activity configuration.

The active I_Offroad build path lives in ``pipeline.py``. This parser remains here
because ``PROXY.visualization.offroad_area_map`` uses it to interpret optional
``area_proxy.subsector_activity`` blocks for map overlays.
"""

from __future__ import annotations

from typing import Any


def parse_offroad_subsector_activity(area_proxy: dict[str, Any]) -> list[dict[str, Any]]:
    raw = area_proxy.get("subsector_activity")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "I_Offroad requires sector_cfg.area_proxy.subsector_activity (non-empty mapping)."
        )
    top_default = area_proxy.get("cams_emission_category_indices")
    default_ec = int(
        area_proxy.get(
            "cams_emission_category_default",
            int(top_default[0]) if isinstance(top_default, list) and top_default else 12,
        )
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(raw.keys()):
        block = raw[key]
        if not isinstance(block, dict):
            raise ValueError(f"area_proxy.subsector_activity[{key!r}] must be a mapping")
        ec = int(block.get("cams_emission_category_index", default_ec))
        codes = block.get("corine_codes")
        if not codes:
            raise ValueError(
                f"area_proxy.subsector_activity[{key!r}] must set corine_codes (list of CLC codes)."
            )
        mass = block.get("emission_mass_fraction")
        vl = block.get("vector_layer")
        rows.append(
            {
                "subsector_key": str(key),
                "nfr2": str(block.get("nfr2", "")).strip(),
                "cams_emission_category_index": ec,
                "corine_codes": tuple(int(x) for x in codes),
                "emission_mass_fraction": float(mass) if mass is not None else None,
                "vector_layer": str(vl).strip() if vl else None,
            }
        )
    return rows


