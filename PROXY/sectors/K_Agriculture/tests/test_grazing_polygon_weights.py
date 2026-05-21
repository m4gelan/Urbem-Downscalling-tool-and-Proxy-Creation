"""Grazing polygon amplification weights (family 3)."""

from __future__ import annotations

import numpy as np

from PROXY.sectors.K_Agriculture.signals.grazing_pasture import polygon_grazing_weight


def test_polygon_grazing_weight_zero_max_count() -> None:
    assert polygon_grazing_weight(0, 0, kappa=0.5, epsilon=1e-6, w_max=4.0) == 1.0


def test_polygon_grazing_weight_scales_with_relative_count() -> None:
    kappa, eps, w_max = 0.5, 1e-6, 4.0
    w0 = polygon_grazing_weight(0, 4, kappa=kappa, epsilon=eps, w_max=w_max)
    w4 = polygon_grazing_weight(4, 4, kappa=kappa, epsilon=eps, w_max=w_max)
    assert w0 == 1.0
    expected_max = min(w_max, 1.0 + kappa * (4.0 / (4.0 + eps)))
    assert np.isclose(w4, expected_max, rtol=1e-9)


def test_polygon_grazing_weight_respects_w_max() -> None:
    w = polygon_grazing_weight(100, 100, kappa=10.0, epsilon=1e-9, w_max=2.5)
    assert w == 2.5
