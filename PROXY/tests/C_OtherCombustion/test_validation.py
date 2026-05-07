from __future__ import annotations

import json
from pathlib import Path

import pytest

from PROXY.sectors.C_OtherCombustion.exceptions import ConfigurationError
from PROXY.sectors.C_OtherCombustion.validation import validate_pipeline_config


def test_validation_raises_on_bad_mapping(tmp_path):
    bad = tmp_path / "GAINS_mapping.json"
    bad.write_text('{"rules": [{"class": "R_FIREPLACE", "fuel_contains": "x"}]}', encoding="utf-8")
    emep = tmp_path / "EMEP.json"
    emep.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "fuel": "Natural gas",
                        "appliance": "Small (single household scale, capacity <=50 kWth) boilers",
                        "ef": {"nox": 42, "nh3": 1},
                    },
                    {
                        "fuel": "Solid fuel (not biomass)",
                        "appliance": "Fireplaces, saunas and outdoor heaters",
                        "ef": {"nox": 60, "nh3": 5},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    side = tmp_path / "side.json"
    side.write_text(
        json.dumps(
            {
                "year": 2021,
                "iso3_to_geo_labels": {"GRC": ["EL"]},
                "class_to_metric": {
                    "R_FIREPLACE": "space_heating",
                    "R_HEATING_STOVE": "space_heating",
                    "R_COOKING_STOVE": "cooking",
                    "R_BOILER_MAN": ["space_heating", "water_heating"],
                    "R_BOILER_AUT": ["space_heating", "water_heating"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "country": {"cams_iso3": "GRC"},
        "eurostat": {"enabled": False},
        "morphology": {
            "urban_111": 111,
            "urban_112": 112,
            "urban_121": 121,
            "residential_fireplace_heating_stove": {"w111": 0.6, "w112": 0.3, "w_other": 0.1},
            "commercial_boilers": {"w111": 0.2, "w121": 0.5, "w_other": 0.3},
        },
    }
    with pytest.raises(ConfigurationError):
        validate_pipeline_config(
            repo_root=tmp_path,
            cfg=cfg,
            mapping_path=bad,
            emep_path=emep,
            sidecar_path=side,
            pollutant_outputs=["nox"],
        )
