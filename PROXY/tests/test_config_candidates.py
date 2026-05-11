"""Ordered YAML/JSON candidate loader (sector base configs)."""

from __future__ import annotations

import warnings

import pytest

from PROXY.core.dataloaders.config_candidates import load_first_existing_yaml_or_json


def test_first_existing_canonical_no_warning(tmp_path) -> None:
    canonical = tmp_path / "rules.yaml"
    canonical.write_text("a: 1\n", encoding="utf-8")
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("b: 2\n", encoding="utf-8")
    out = load_first_existing_yaml_or_json(
        [canonical, legacy],
        context="test",
    )
    assert out == {"a": 1}


def test_legacy_second_emits_deprecation(tmp_path) -> None:
    canonical = tmp_path / "rules.yaml"
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("x: 9\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = load_first_existing_yaml_or_json(
            [canonical, legacy],
            context="test",
        )
    assert out == {"x": 9}
    assert len(rec) == 1
    assert issubclass(rec[0].category, DeprecationWarning)
    assert "legacy" in str(rec[0].message).lower()


def test_none_exist_raises() -> None:
    with pytest.raises(FileNotFoundError, match="none of the candidate"):
        load_first_existing_yaml_or_json([], context="empty")
