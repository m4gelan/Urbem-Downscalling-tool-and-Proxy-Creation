"""Method 1 unified ``pool`` key (EU27 string vs country list)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from PROXY.core.alpha.alpha_method_engine import build_alpha_tensor_methods


def _tiny_long_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for iso, g1, g2, g3 in [
        ("EU27", 90.0, 9.0, 1.0),
        ("FRA", 10.0, 30.0, 60.0),
        ("DEU", 50.0, 25.0, 25.0),
    ]:
        for g, e in (("G1", g1), ("G2", g2), ("G3", g3)):
            rows.append(
                {
                    "country_iso3": iso,
                    "pollutant": "nox",
                    "group": g,
                    "E": e,
                }
            )
    return pd.DataFrame(rows)


def test_pool_country_list_mean_not_eu27_row() -> None:
    doc = {
        "sectors": {
            "I_Offroad": {
                "gnfr_letter": "I",
                "defaults": {"method": 1, "pool": ["FRA", "DEU"]},
                "pollutants": {"nox": {"method": 1}},
            }
        }
    }
    cfg = {
        "_project_root": Path(__file__).resolve().parents[2],
        "_alpha_methods_doc_override": doc,
    }

    long_df = _tiny_long_df()
    iso3_list = ["GRC"]
    pollutants = ["nox"]
    group_order = ("G1", "G2", "G3")

    alpha, _, _ = build_alpha_tensor_methods(
        long_df,
        iso3_list,
        pollutants,
        group_order,
        sector_key="I_Offroad",
        cfg=cfg,
    )

    fra = np.array([10.0, 30.0, 60.0]) / 100.0
    deu = np.array([50.0, 25.0, 25.0]) / 100.0
    expected = (fra + deu) / 2.0
    expected = expected / expected.sum()

    eu_row = np.array([90.0, 9.0, 1.0]) / 100.0

    got = alpha[0, :, 0]
    assert np.allclose(got, expected, atol=1e-9)
    assert not np.allclose(got, eu_row, atol=1e-4)
