from __future__ import annotations

from pathlib import Path

import numpy as np

from PROXY.sectors.C_OtherCombustion.constants import MODEL_CLASSES
from PROXY.sectors.C_OtherCombustion.m_builder.assemble import build_m_matrix
from PROXY.sectors.C_OtherCombustion.m_builder.emep_ef import load_emep
from PROXY.sectors.C_OtherCombustion.m_builder.enduse_factors import EndUseFactors
from PROXY.sectors.C_OtherCombustion.m_builder.gains_activity import index_gains_files
from PROXY.sectors.C_OtherCombustion.m_builder.mapping_io import load_gains_mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_disabled_eurostat_matches_legacy_scalars():
    root = _repo_root()
    gains_dir = root / "INPUT" / "Proxy" / "ProxySpecific" / "OtherCombustion" / "GAINS"
    if not gains_dir.is_dir():
        return
    mapping_path = root / "PROXY" / "config" / "other_combustion" / "GAINS_mapping.yaml"
    emep_path = root / "PROXY" / "config" / "other_combustion" / "EMEP_emission_factors.yaml"
    if not mapping_path.is_file() or not emep_path.is_file():
        return
    rules, hints = load_gains_mapping(mapping_path)
    emep = load_emep(emep_path)
    idx = index_gains_files(gains_dir, {}, root)
    if "DEU" not in idx:
        return
    gp = idx["DEU"]
    pols = ["nh3", "nox"]
    legacy = {k: 1.0 for k in MODEL_CLASSES}
    M_legacy = build_m_matrix(
        gp,
        "2020",
        rules,
        EndUseFactors.from_legacy_class_scalars(legacy),
        emep,
        pols,
        emep_fuel_hints=hints,
    )
    M_new = build_m_matrix(
        gp,
        "2020",
        rules,
        EndUseFactors.disabled_uniform(),
        emep,
        pols,
        emep_fuel_hints=hints,
    )
    np.testing.assert_allclose(M_legacy, M_new, rtol=0, atol=1e-9)
