"""Lock behaviour of inventory sector / NFR token normalization."""

from __future__ import annotations

import pytest

from PROXY.core.alpha.reported import normalize_inventory_sector, normalize_sector_token


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1A1a", "1A1A"),
        ("  2.B.5  ", "2B5"),
        ("NFR-3", "NFR3"),
        ("", ""),
        ("a-b_c.d", "ABCD"),
    ],
)
def test_normalize_inventory_sector_matches_strip_upper_alnum(raw: str, expected: str) -> None:
    assert normalize_inventory_sector(raw) == expected


def test_normalize_inventory_sector_accepts_non_str() -> None:
    assert normalize_inventory_sector(12) == "12"


def test_normalize_inventory_sector_alias_of_sector_token_for_strings() -> None:
    s = "  1A4  "
    assert normalize_inventory_sector(s) == normalize_sector_token(s)
