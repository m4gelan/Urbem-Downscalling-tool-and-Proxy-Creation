"""Beta mix: nine subsector raw proxies from four archetypes."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_beta(beta: dict[str, dict[str, float]], arche_keys: tuple[str, ...]) -> list[str]:
    errs: list[str] = []
    for sub, row in beta.items():
        s = sum(float(row.get(k, 0.0)) for k in arche_keys)
        if abs(s - 1.0) > 1e-6:
            errs.append(f"beta row {sub!r} sums to {s}, expected 1")
    return errs


def build_subsector_raw_stack(
    arche: dict[str, np.ndarray],
    beta: dict[str, dict[str, float]],
    subsectors: list[str],
) -> np.ndarray:
    """Shape (H, W, S) float32."""
    h, w = next(iter(arche.values())).shape
    S = len(subsectors)
    stack = np.zeros((h, w, S), dtype=np.float32)
    ak = ("house", "serv", "ind", "infra")
    z = np.zeros((h, w), dtype=np.float32)
    for si, sub in enumerate(subsectors):
        row = beta.get(sub) or {}
        acc = np.zeros((h, w), dtype=np.float32)
        for k in ak:
            acc += float(row.get(k, 0.0)) * np.maximum(arche.get(k, z), 0.0)
        stack[:, :, si] = acc.astype(np.float32)
    return stack
