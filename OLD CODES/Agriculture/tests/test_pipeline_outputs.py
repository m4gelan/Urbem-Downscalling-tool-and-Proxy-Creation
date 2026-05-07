from __future__ import annotations

import pandas as pd

from Agriculture.config import load_alpha_config
from Agriculture.core.pipeline import _alpha_pollutants, _process_ids_needed, _sanitize_token
from Agriculture.core.run_countries import parse_run_country_codes
from Agriculture.source_relevance.agricultural_soils import nmvoc_score_per_point


def test_alpha_process_resolution_includes_pm_and_housing() -> None:
    alpha = load_alpha_config()
    pollutants = _alpha_pollutants(alpha)
    assert "PM10" in pollutants
    assert "PM2.5" in pollutants
    process_ids = _process_ids_needed(alpha)
    assert "livestock_housing" in process_ids
    assert "biomass_burning" in process_ids


def test_sanitize_token_handles_decimal_pollutants() -> None:
    assert _sanitize_token("PM2.5") == "PM2_5"
    assert _sanitize_token("NOx") == "NOx"


def test_b33_is_non_emitting_for_crop_nmvoc() -> None:
    row = pd.Series({"SURVEY_LC1": "B33", "SURVEY_LC1_PERC": 100})
    assert nmvoc_score_per_point(row) == 0.0


def test_parse_run_country_codes_variants() -> None:
    assert parse_run_country_codes({}) is None
    assert parse_run_country_codes({"country": ""}) is None
    assert parse_run_country_codes({"country": "DE"}) == frozenset({"DE"})
    assert parse_run_country_codes({"country": "DE, FR, EL"}) == frozenset({"DE", "FR", "EL"})
    assert parse_run_country_codes({"country": ["DE", "FR"]}) == frozenset({"DE", "FR"})
