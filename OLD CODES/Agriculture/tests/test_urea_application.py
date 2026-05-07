"""Urea proxy: cropland score 1.0, grassland 0.7*omega, empty aggregation -> mu 0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Agriculture.source_relevance.common import aggregate_nuts_clc_mu, merge_extent_mu_rho
from Agriculture.source_relevance.urea_application import (
    grass_livestock_omega,
    point_urea_score,
)


def test_grass_livestock_omega_no_clip_and_median():
    idx = pd.Index(["A", "B", "C"])
    i = pd.Series([0.0, 4.0, 10.0], index=idx)
    w = grass_livestock_omega(i)
    med = 7.0
    np.testing.assert_allclose(w["A"], 0.0)
    np.testing.assert_allclose(w["B"], 4.0 / med)
    np.testing.assert_allclose(w["C"], 10.0 / med)


def test_grass_livestock_omega_fallback_all_ones_when_no_positive():
    i = pd.Series([0.0, 0.0], index=["x", "y"])
    w = grass_livestock_omega(i)
    assert (w == 1.0).all()


def test_point_urea_cropland_is_one():
    omega = pd.Series([2.0], index=["DE111"])
    s = point_urea_score("B11", "U111", "DE111", omega)
    assert abs(s - 1.0) < 1e-12


def test_point_urea_grassland_is_07_times_omega():
    omega = pd.Series([2.0], index=["DE111"])
    s = point_urea_score("E10", "U111", "DE111", omega)
    assert abs(s - 1.4) < 1e-12


def test_point_urea_missing_nuts_uses_fallback_omega():
    omega = pd.Series([3.0], index=["OTHER"])
    s = point_urea_score("E10", "U111", "DE999", omega)
    assert abs(s - 0.7 * 1.0) < 1e-12


def test_empty_cell_mu_zero_after_merge():
    agg = pd.DataFrame(
        {
            "NUTS_ID": ["N1"],
            "CLC_CODE": [18],
            "COUNTRY": ["XX"],
            "mu": [0.5],
            "rho": [1.0],
        }
    )
    extent_df = pd.DataFrame(
        {
            "NUTS_ID": ["N1", "N1"],
            "CLC_CODE": [18, 12],
            "COUNTRY": ["XX", "XX"],
            "n_pixels": [10, 5],
        }
    )
    out = merge_extent_mu_rho(extent_df, agg)
    row12 = out[out["CLC_CODE"] == 12].iloc[0]
    assert row12["mu"] == 0.0
    assert row12["rho"] == 0.0


def test_aggregate_skips_nan_points():
    pts = pd.DataFrame(
        {
            "NUTS_ID": ["N1", "N1", "N1"],
            "CLC_CODE": [18, 18, 18],
            "COUNTRY": ["XX", "XX", "XX"],
            "mu": [1.0, float("nan"), 0.4],
        }
    )
    g = aggregate_nuts_clc_mu(pts, "mu")
    assert len(g) == 1
    assert abs(g.iloc[0]["mu"] - 0.7) < 1e-9
