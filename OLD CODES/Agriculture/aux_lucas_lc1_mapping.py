"""
Minimal LUCAS LC1 -> crop category mapping (for Auxiliaries scripts only).

Extracted from the former lucas_relevance_build module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LUCAS_LC1_TO_CROP: dict[str, str] = {
    "B11": "wheat",
    "B12": "wheat",
    "B13": "barley",
    "B14": "rye",
    "B15": "oats",
    "B16": "maize_grain",
    "B17": "rice",
    "B18": "triticale",
    "B19": "other_cereal",
    "B21": "potato",
    "B22": "sugar_beet",
    "B23": "other_root",
    "B31": "sunflower",
    "B32": "rapeseed",
    "B33": "soya",
    "B34": "cotton",
    "B35": "other_fiber",
    "B36": "tobacco",
    "B37": "other_fiber",
    "B41": "dry_pulse",
    "B42": "tomato",
    "B43": "other_veg",
    "B44": "flowers",
    "B45": "strawberry",
    "B51": "fodder_legume",
    "B52": "fodder_legume",
    "B53": "fodder_legume",
    "B54": "fodder_legume",
    "B55": "Temporary_grass",
    "B71": "fruit_tree",
    "B72": "fruit_tree",
    "B73": "fruit_tree",
    "B74": "nuts_tree",
    "B75": "fruit_tree",
    "B76": "citrus",
    "B77": "citrus",
    "B81": "olive_tree",
    "B82": "vineyards",
    "B83": "nurseries",
    "B84": "permanenet_industrial",
    "BX1": "fallow",
    "BX2": "fallow",
    "E10": "permanent_grass",
    "E20": "permanent_grass",
    "E30": "shrub_grass",
    "B": None,
    "E": "permanent_grass",
}


def _normalize_code(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = str(s).strip().strip('"').strip("'").strip()
    t = " ".join(t.split()).upper()
    if t in ("NAN", "NONE", "NA", "#N/A", "NAT"):
        return ""
    return t


def crop_category_from_lc1(lc1: Any) -> str | None:
    code = _normalize_code(lc1)
    if not code:
        return None
    return LUCAS_LC1_TO_CROP.get(code)
