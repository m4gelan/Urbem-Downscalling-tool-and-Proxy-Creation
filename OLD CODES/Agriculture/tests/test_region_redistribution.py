from __future__ import annotations

import pandas as pd
import pytest

from Agriculture.analysis.weights.run_weight_analysis import _build_region_redistribution_from_long


def test_build_region_redistribution_from_long_keeps_selected_regions_and_class_shares() -> None:
    long_df = pd.DataFrame(
        [
            {
                "pollutant": "NH3",
                "NUTS_ID": "R1",
                "CLC_CODE": 12,
                "NAME_REGION": "Region 1",
                "COUNTRY": "AA",
                "n_pixels": 80,
                "w_p": 0.70,
            },
            {
                "pollutant": "NH3",
                "NUTS_ID": "R1",
                "CLC_CODE": 18,
                "NAME_REGION": "Region 1",
                "COUNTRY": "AA",
                "n_pixels": 20,
                "w_p": 0.30,
            },
            {
                "pollutant": "NOx",
                "NUTS_ID": "R1",
                "CLC_CODE": 12,
                "NAME_REGION": "Region 1",
                "COUNTRY": "AA",
                "n_pixels": 80,
                "w_p": 0.75,
            },
            {
                "pollutant": "NOx",
                "NUTS_ID": "R1",
                "CLC_CODE": 18,
                "NAME_REGION": "Region 1",
                "COUNTRY": "AA",
                "n_pixels": 20,
                "w_p": 0.25,
            },
            {
                "pollutant": "NH3",
                "NUTS_ID": "R2",
                "CLC_CODE": 12,
                "NAME_REGION": "Region 2",
                "COUNTRY": "BB",
                "n_pixels": 50,
                "w_p": 1.00,
            },
        ]
    )
    selection_df = pd.DataFrame(
        [
            {
                "country": "AA",
                "COUNTRY": "AA",
                "NUTS_ID": "R1",
                "NAME_REGION": "Region 1",
                "selected_clc_code": 12,
            }
        ]
    )

    out = _build_region_redistribution_from_long(long_df, selection_df)

    assert set(out["NUTS_ID"]) == {"R1"}
    assert set(out["pollutant"]) == {"NH3", "NOx"}
    assert out.groupby("pollutant")["class_share"].sum().round(10).eq(1.0).all()
    nh3_12 = out[(out["pollutant"] == "NH3") & (out["CLC_CODE"] == 12)].iloc[0]
    assert nh3_12["baseline_w"] == 0.8
    assert nh3_12["class_share"] == 0.8
    assert nh3_12["delta"] == pytest.approx(-0.1)
