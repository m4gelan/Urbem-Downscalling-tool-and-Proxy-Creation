from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

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
    nox_path = bundle_dir / manifest["cams_nox_file"]
    nox_by_cell = {int(k): float(v) for k, v in json.loads(nox_path.read_text(encoding="utf-8")).items()}

    return {
        "manifest": manifest,
        "cell_id": cell_id,
        "W_by_group": W_by_group,
        "group_names": [str(ent["name"]) for ent in manifest["groups"]],
        "alpha": alpha_doc,
        "nox_by_cell": nox_by_cell,
        "bundle_dir": bundle_dir,
        "has_mix": bool(manifest.get("mix")),
    }
