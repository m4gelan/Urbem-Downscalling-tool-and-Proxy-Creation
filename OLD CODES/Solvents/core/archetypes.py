"""Combine normalized indicators into four archetype proxies + optional generic blend."""

from __future__ import annotations

from typing import Any

import numpy as np

from .normalize import normalize_indicator


def _combine_weighted(
    parts: dict[str, np.ndarray],
    weights: dict[str, float],
    norm_cfg: dict[str, Any],
) -> np.ndarray:
    """Renormalize weights over keys present in parts; skip missing keys."""
    keys = [k for k in weights if k in parts and parts[k] is not None]
    if not keys:
        h, w = next(iter(parts.values())).shape
        return np.zeros((h, w), dtype=np.float32)
    wts = {k: float(weights[k]) for k in keys}
    s = sum(wts.values())
    if s <= 0:
        h, w = next(iter(parts.values())).shape
        return np.zeros((h, w), dtype=np.float32)
    wts = {k: wts[k] / s for k in keys}
    acc = None
    for k in keys:
        z = normalize_indicator(parts[k], **norm_cfg)
        contrib = wts[k] * z
        acc = contrib if acc is None else acc + contrib
    return np.maximum(acc.astype(np.float32), 0.0)


def build_archetype_proxies(
    raw: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    raw: named float32 (H,W) non-negative indicator fields.
    Returns rho_house, rho_serv, rho_ind, rho_infra.
    """
    norm_cfg = dict(cfg.get("normalization") or {})
    aw = cfg.get("archetype_weights") or {}

    house = _combine_weighted(
        {k: raw[k] for k in ("population", "residential_share") if k in raw},
        aw.get("house") or {},
        norm_cfg,
    )
    serv = _combine_weighted(
        {
            k: raw[k]
            for k in ("urban_fabric", "service_land", "service_osm")
            if k in raw
        },
        aw.get("serv") or {},
        norm_cfg,
    )
    ind = _combine_weighted(
        {
            k: raw[k]
            for k in ("industry_clc", "industry_osm", "industry_buildings")
            if k in raw
        },
        aw.get("ind") or {},
        norm_cfg,
    )
    infra = _combine_weighted(
        {
            k: raw[k]
            for k in (
                "road_length",
                "weighted_road_length",
                "transport_area",
                "roof_area",
            )
            if k in raw
        },
        aw.get("infra") or {},
        norm_cfg,
    )
    return {
        "house": house,
        "serv": serv,
        "ind": ind,
        "infra": infra,
    }


def generic_proxy(arche: dict[str, np.ndarray], omega: dict[str, float]) -> np.ndarray:
    keys = [k for k in omega if k in arche]
    s = sum(float(omega[k]) for k in keys)
    if s <= 0:
        h, w = next(iter(arche.values())).shape
        return np.zeros((h, w), dtype=np.float32)
    acc = np.zeros_like(next(iter(arche.values())), dtype=np.float32)
    for k in keys:
        acc += (float(omega[k]) / s) * np.maximum(arche[k], 0.0).astype(np.float32)
    return acc
