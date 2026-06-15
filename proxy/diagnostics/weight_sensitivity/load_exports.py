from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

from proxy.diagnostics.weight_sensitivity.pollutants_config import REFERENCE_POLLUTANTS

MULTI_GROUP_SECTORS = (
    "B_Industry",
    "D_Fugitive",
    "E_Solvents",
    "I_Offroad",
    "J_Waste",
    "K_Agriculture",
)

PRONG_B_REP_SECTORS = ("A_PublicPower", "B_Industry", "J_Waste")


def bundle_path(export_root: Path, sector_key: str, country_tag: str, year: int) -> Path:
    return export_root / sector_key / f"{country_tag}_{int(year)}"


def _parse_mass_json(doc: dict[str, Any]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for pol, cells in doc.items():
        if not isinstance(cells, dict):
            continue
        out[str(pol)] = {int(k): float(v) for k, v in cells.items()}
    return out


def load_mass_by_pollutant(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[int, float]]:
    mass_file = manifest.get("cams_mass_file")
    if mass_file:
        path = bundle_dir / str(mass_file)
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                return _parse_mass_json(doc)
    nox_path = bundle_dir / manifest["cams_nox_file"]
    nox_by_cell = {int(k): float(v) for k, v in json.loads(nox_path.read_text(encoding="utf-8")).items()}
    legacy = str(manifest.get("reference_pollutant") or "NOx")
    return {legacy: nox_by_cell}


def mass_for_pollutant(
    mass_by_pollutant: dict[str, dict[int, float]],
    pollutant: str,
) -> dict[int, float]:
    return dict(mass_by_pollutant.get(pollutant) or {})


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"W_groups bundle missing: {bundle_dir}")
    with (bundle_dir / "groups_manifest.yaml").open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid manifest: {bundle_dir / 'groups_manifest.yaml'}")

    with rasterio.open(bundle_dir / manifest["cell_id_file"]) as src:
        cell_id = src.read(1).astype(np.int32)

    W_by_group: dict[str, np.ndarray] = {}
    for ent in manifest["groups"]:
        gname = str(ent["name"])
        with rasterio.open(bundle_dir / str(ent["file"])) as src:
            W_by_group[gname] = src.read(1).astype(np.float32)

    alpha_path = bundle_dir / manifest["alpha_matrix_file"]
    alpha_doc = json.loads(alpha_path.read_text(encoding="utf-8"))
    mass_by_pollutant = load_mass_by_pollutant(bundle_dir, manifest)
    ref_pollutants = manifest.get("reference_pollutants") or list(REFERENCE_POLLUTANTS)
    nox_by_cell = mass_for_pollutant(mass_by_pollutant, "NOx")
    if not nox_by_cell and mass_by_pollutant:
        nox_by_cell = mass_for_pollutant(mass_by_pollutant, next(iter(mass_by_pollutant)))

    return {
        "manifest": manifest,
        "cell_id": cell_id,
        "W_by_group": W_by_group,
        "group_names": [str(ent["name"]) for ent in manifest["groups"]],
        "alpha": alpha_doc,
        "mass_by_pollutant": mass_by_pollutant,
        "reference_pollutants": [str(p) for p in ref_pollutants],
        "nox_by_cell": nox_by_cell,
        "bundle_dir": bundle_dir,
        "has_mix": bool(manifest.get("mix")),
    }
