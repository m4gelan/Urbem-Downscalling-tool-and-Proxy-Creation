"""
pytest: livestock housing score ordering g=2 >= g=0 >= g=1 for fixed (LU1, LC1).
"""

from __future__ import annotations

import pandas as pd

from Agriculture.source_relevance.livestock_housing import point_livestock_housing_nh3


def _row(graze: float, lc: str, lu: str) -> pd.Series:
    return pd.Series({"SURVEY_GRAZING": graze, "SURVEY_LC1": lc, "SURVEY_LU1": lu})


def test_grazing_score_ordering() -> None:
    for lu, lc in [
        ("U111", "E10"),
        ("U111", "B11"),
        ("U112", "E10"),
        ("U111", "A11"),
    ]:
        s2 = point_livestock_housing_nh3(_row(2.0, lc, lu))
        s0 = point_livestock_housing_nh3(_row(0.0, lc, lu))
        s1 = point_livestock_housing_nh3(_row(1.0, lc, lu))
        assert s1 == 0.0
        assert s2 is not None and s0 is not None
        assert s2 >= s0 >= s1


def test_forestry_and_kitchen_garden_excluded() -> None:
    assert point_livestock_housing_nh3(_row(2.0, "B11", "U120")) is None
    assert point_livestock_housing_nh3(_row(2.0, "B81", "U113")) is None


def test_tier_scores_examples() -> None:
    assert abs(point_livestock_housing_nh3(_row(2.0, "E10", "U111")) - 0.55) < 1e-9
    assert abs(point_livestock_housing_nh3(_row(2.0, "B11", "U111")) - 0.20) < 1e-9
    assert abs(point_livestock_housing_nh3(_row(float("nan"), "E10", "U111")) - 0.22) < 1e-9
