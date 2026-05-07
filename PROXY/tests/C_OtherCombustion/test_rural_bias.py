from __future__ import annotations

import numpy as np

from PROXY.sectors.C_OtherCombustion.x_builder.rural_bias import rural_bias_from_density


def test_rural_bias_bounds():
    d = np.array([[0.0, 1.0], [10.0, 100.0]], dtype=np.float32)
    rb = rural_bias_from_density(d, rural_min=0.4)
    assert rb.shape == d.shape
    assert float(np.min(rb)) >= 0.4 - 1e-5
    assert float(np.max(rb)) <= 1.0 + 1e-5
