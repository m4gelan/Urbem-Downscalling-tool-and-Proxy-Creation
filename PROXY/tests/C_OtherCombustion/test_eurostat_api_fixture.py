from __future__ import annotations

import json
from pathlib import Path

import pytest

from PROXY.sectors.C_OtherCombustion.eurostat_api import parse_nrg_bal_s_commercial_alpha, parse_nrg_d_hhq_metric_tj


@pytest.fixture()
def minimal_nrg_bal_s() -> dict:
    return {
        "dataset": {
            "dimension": {
                "nrg_bal": {"category": {"index": {"FC_OTH_CP_E": 0, "FC_OTH_HH_E": 1}}},
                "unit": {"category": {"index": {"TJ": 0}}},
                "geo": {"category": {"index": {"EL": 0}}},
                "time": {"category": {"index": {"2021": 0}}},
            },
            "id": ["nrg_bal", "unit", "geo", "time"],
            "size": [2, 1, 1, 1],
            "value": [200.0, 800.0],
        }
    }


@pytest.fixture()
def minimal_nrg_d_hhq() -> dict:
    return {
        "dataset": {
            "dimension": {
                "nrg_bal": {
                    "category": {
                        "index": {
                            "FC_OTH_HH_E_SH": 0,
                            "FC_OTH_HH_E_WH": 1,
                            "FC_OTH_HH_E_CK": 2,
                            "FC_OTH_HH_E_CL": 3,
                            "FC_OTH_HH_E_LE": 4,
                            "FC_OTH_HH_E_OE": 5,
                        }
                    }
                },
                "unit": {"category": {"index": {"TJ": 0}}},
                "siec": {"category": {"index": {"TOTAL": 0}}},
                "geo": {"category": {"index": {"EL": 0}}},
                "time": {"category": {"index": {"2021": 0}}},
            },
            "id": ["nrg_bal", "unit", "siec", "geo", "time"],
            "size": [6, 1, 1, 1, 1],
            "value": [500.0, 200.0, 100.0, 50.0, 100.0, 50.0],
        }
    }


def test_parse_commercial_alpha(minimal_nrg_bal_s):
    a, src = parse_nrg_bal_s_commercial_alpha(minimal_nrg_bal_s, geo="EL", year=2021)
    assert src == "api"
    assert a is not None
    assert abs(a - 0.2) < 1e-9


def test_parse_hhq_metrics(minimal_nrg_d_hhq):
    m, src = parse_nrg_d_hhq_metric_tj(minimal_nrg_d_hhq, geo="EL", year=2021)
    assert src == "api"
    assert m["space_heating"] == 500.0
    assert len(m) == 6


def test_fixture_roundtrip(tmp_path, minimal_nrg_d_hhq):
    p = tmp_path / "nrg_d_hhq_EL_2021.json"
    p.write_text(json.dumps(minimal_nrg_d_hhq), encoding="utf-8")
    root = json.loads(p.read_text(encoding="utf-8"))
    m, _ = parse_nrg_d_hhq_metric_tj(root, geo="EL", year=2021)
    assert sum(m.values()) == 1000.0


@pytest.fixture()
def nrg_bal_s_sparse_freq_siec() -> dict:
    """Eurostat JSON-stat 2.0: ``value`` as sparse dict; ``freq``/``siec`` in ``id`` order."""
    return {
        "dataset": {
            "dimension": {
                "freq": {"category": {"index": {"A": 0}}},
                "nrg_bal": {"category": {"index": {"FC_OTH_CP_E": 0, "FC_OTH_HH_E": 1}}},
                "siec": {"category": {"index": {"TOTAL": 0}}},
                "unit": {"category": {"index": {"TJ": 0}}},
                "geo": {"category": {"index": {"EL": 0}}},
                "time": {"category": {"index": {"2021": 0}}},
            },
            "id": ["freq", "nrg_bal", "siec", "unit", "geo", "time"],
            "size": [1, 2, 1, 1, 1, 1],
            "value": {"0": 200.0, "1": 800.0},
        }
    }


def test_parse_commercial_alpha_sparse_dict(nrg_bal_s_sparse_freq_siec):
    a, src = parse_nrg_bal_s_commercial_alpha(nrg_bal_s_sparse_freq_siec, geo="EL", year=2021)
    assert src == "api"
    assert a is not None
    assert abs(a - 0.2) < 1e-9


def test_parse_hhq_single_balance_sparse_value():
    root = {
        "dataset": {
            "dimension": {
                "freq": {"category": {"index": {"A": 0}}},
                "nrg_bal": {"category": {"index": {"FC_OTH_HH_E_SH": 0}}},
                "unit": {"category": {"index": {"TJ": 0}}},
                "siec": {"category": {"index": {"TOTAL": 0}}},
                "geo": {"category": {"index": {"EL": 0}}},
                "time": {"category": {"index": {"2021": 0}}},
            },
            "id": ["freq", "nrg_bal", "unit", "siec", "geo", "time"],
            "size": [1, 1, 1, 1, 1, 1],
            "value": {"0": 123.45},
        }
    }
    m, src = parse_nrg_d_hhq_metric_tj(root, geo="EL", year=2021)
    assert src == "api"
    assert m == {"space_heating": pytest.approx(123.45)}
