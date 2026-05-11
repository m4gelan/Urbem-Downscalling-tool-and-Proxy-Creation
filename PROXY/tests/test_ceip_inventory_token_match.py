"""CEIP inventory token -> GNFR group id uses longest-prefix semantics."""

from __future__ import annotations

from PROXY.core.alpha.reported_group_alpha import _group_id_for_ceip_inventory_token


def test_longest_nfr_prefix_wins() -> None:
    tok2g = {"1A": "G1", "1A1": "G2"}
    assert _group_id_for_ceip_inventory_token("1A1A", tok2g) == "G2"
    assert _group_id_for_ceip_inventory_token("1A9", tok2g) == "G1"


def test_exact_match() -> None:
    tok2g = {"1A1": "G2", "1A": "G1"}
    assert _group_id_for_ceip_inventory_token("1A1", tok2g) == "G2"


def test_offroad_1a3eii_maps_to_g3_not_pipeline_prefix() -> None:
    """1A3eii must not match the shorter prefix 1A3ei (GNFR I triple-leg)."""
    from PROXY.core.alpha.reported_group_alpha import _sector_to_group

    tok2g = {
        "1A3C": "G1",
        "1A3EI": "G2",
        "1A3EII": "G3",
    }
    assert _sector_to_group("1A3EII", tok2g) == "G3"
    assert _sector_to_group("1A3EI", tok2g) == "G2"


def test_no_match_returns_none() -> None:
    assert _group_id_for_ceip_inventory_token("9Z", {"1A": "G1"}) is None
