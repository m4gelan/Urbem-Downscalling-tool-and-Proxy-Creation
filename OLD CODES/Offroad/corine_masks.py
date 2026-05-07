"""CORINE code masks on the reference grid."""

from __future__ import annotations

import numpy as np

from PROXY.core.osm_corine_proxy import adapt_corine_classes_for_grid


def corine_binary_mask(clc_nn: np.ndarray, codes: list[int]) -> np.ndarray:
    """Max indicator over mutually exclusive CLC codes."""
    ci = np.asarray(clc_nn, dtype=np.int32)
    acc = np.zeros(ci.shape, dtype=np.float32)
    for c in codes:
        acc = np.maximum(acc, (ci == int(c)).astype(np.float32))
    return acc


def corine_binary_mask_adapted(
    clc_nn: np.ndarray,
    yaml_codes: list[int],
) -> tuple[np.ndarray, list[int]]:
    codes_adapted, _remapped = adapt_corine_classes_for_grid(clc_nn, [int(x) for x in yaml_codes])
    if not codes_adapted:
        codes_adapted = [int(x) for x in yaml_codes]
    acc = np.zeros(np.asarray(clc_nn).shape, dtype=np.float32)
    ci = np.asarray(clc_nn, dtype=np.int32)
    for c in codes_adapted:
        acc = np.maximum(acc, (ci == int(c)).astype(np.float32))
    return acc, codes_adapted
