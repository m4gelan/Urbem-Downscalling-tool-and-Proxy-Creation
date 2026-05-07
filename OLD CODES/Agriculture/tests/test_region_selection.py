from __future__ import annotations

import pandas as pd

from Agriculture.analysis.weights.region_selection import build_region_selection_from_tables


def test_build_region_selection_from_tables_picks_two_regions_per_clc() -> None:
    wide_df = pd.DataFrame(
        [
            {"NUTS_ID": "R1", "COUNTRY": "AA", "NAME_REGION": "Region 1", "CLC_CODE": 12, "n_pixels": 90},
            {"NUTS_ID": "R1", "COUNTRY": "AA", "NAME_REGION": "Region 1", "CLC_CODE": 18, "n_pixels": 10},
            {"NUTS_ID": "R2", "COUNTRY": "AA", "NAME_REGION": "Region 2", "CLC_CODE": 12, "n_pixels": 80},
            {"NUTS_ID": "R2", "COUNTRY": "AA", "NAME_REGION": "Region 2", "CLC_CODE": 18, "n_pixels": 20},
            {"NUTS_ID": "R3", "COUNTRY": "BB", "NAME_REGION": "Region 3", "CLC_CODE": 12, "n_pixels": 20},
            {"NUTS_ID": "R3", "COUNTRY": "BB", "NAME_REGION": "Region 3", "CLC_CODE": 18, "n_pixels": 80},
            {"NUTS_ID": "R4", "COUNTRY": "BB", "NAME_REGION": "Region 4", "CLC_CODE": 12, "n_pixels": 10},
            {"NUTS_ID": "R4", "COUNTRY": "BB", "NAME_REGION": "Region 4", "CLC_CODE": 18, "n_pixels": 90},
        ]
    )
    process_df = pd.DataFrame(
        [
            {"NUTS_ID": "R1", "CLC_CODE": 12, "mu": 2.0, "rho": 0.9},
            {"NUTS_ID": "R1", "CLC_CODE": 18, "mu": 1.0, "rho": 0.4},
            {"NUTS_ID": "R2", "CLC_CODE": 12, "mu": 1.8, "rho": 0.8},
            {"NUTS_ID": "R2", "CLC_CODE": 18, "mu": 1.1, "rho": 0.5},
            {"NUTS_ID": "R3", "CLC_CODE": 12, "mu": 1.1, "rho": 0.5},
            {"NUTS_ID": "R3", "CLC_CODE": 18, "mu": 1.9, "rho": 0.8},
            {"NUTS_ID": "R4", "CLC_CODE": 12, "mu": 1.0, "rho": 0.4},
            {"NUTS_ID": "R4", "CLC_CODE": 18, "mu": 2.1, "rho": 0.9},
        ]
    )

    out = build_region_selection_from_tables(
        wide_df,
        process_df,
        ["AA", "BB"],
        process_id="fertilized_land",
        regions_per_clc=2,
    )

    assert set(out["selected_clc_code"]) == {12, 18}
    assert (out.groupby("selected_clc_code").size() == 2).all()
    assert out["NUTS_ID"].nunique() == 4
    assert set(out["selection"]) == {"representative"}
