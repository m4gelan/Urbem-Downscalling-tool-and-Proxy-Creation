"""CAMS-cell renormalization for combined agriculture proxies."""

from __future__ import annotations

import numpy as np

from PROXY.sectors.K_Agriculture.rasterize_kl import _cams_normalize_raw_to_weights


def test_cams_normalize_constant_raw_sums_to_one_per_cell() -> None:
    h, w = 4, 4
    cell_of = np.zeros((h, w), dtype=np.int32)
    cell_of[:, :2] = 0
    cell_of[:, 2:] = 1
    m_kl = np.array([1, 1], dtype=np.uint8)
    in_ag = np.ones((h, w), dtype=bool)
    raw = np.ones((h, w), dtype=np.float32)

    out = _cams_normalize_raw_to_weights(raw, cell_of, m_kl, in_ag)

    for ci in (0, 1):
        m = cell_of == ci
        assert np.isclose(float(out[m].sum()), 1.0, rtol=1e-5)
        assert np.allclose(out[m], 1.0 / float(np.count_nonzero(m)))


def test_cams_normalize_uniform_fallback_when_cell_sum_zero() -> None:
    h, w = 2, 4
    cell_of = np.full((h, w), 0, dtype=np.int32)
    m_kl = np.array([1, 0], dtype=np.uint8)
    in_ag = np.ones((h, w), dtype=bool)
    raw = np.zeros((h, w), dtype=np.float32)

    out = _cams_normalize_raw_to_weights(raw, cell_of, m_kl, in_ag)

    ag_in_cell = (cell_of == 0) & in_ag
    n = int(np.count_nonzero(ag_in_cell))
    assert n == h * w
    assert np.allclose(out[ag_in_cell], 1.0 / n)
