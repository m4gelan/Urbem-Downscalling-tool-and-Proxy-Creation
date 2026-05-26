from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def sidecar_path(link_path: Path) -> Path:
    return link_path.with_name(f"{link_path.stem}_matches.json")


def load_match_sidecar(link_path: Path) -> dict[str, dict[str, Any]]:
    path = sidecar_path(link_path)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def sidecar_has_emissions(sidecar: dict[str, dict[str, Any]]) -> bool:
    return any(isinstance(rec.get("pollutants"), dict) for rec in sidecar.values())


def meta_for_cams(sidecar: dict[str, dict[str, Any]], cams_point_id: int) -> dict[str, Any]:
    return dict(sidecar.get(str(cams_point_id)) or {})


def flatten_meta_to_row(meta: dict[str, Any]) -> dict[str, Any]:
    if not meta:
        return {}
    out = {
        "match_dataset": meta.get("dataset"),
        "match_dataset_key": meta.get("dataset_key"),
        "facility_name": meta.get("facility_name"),
        "facility_id": meta.get("facility_id"),
    }
    if meta.get("match_distance_km") is not None:
        out["match_distance_km"] = meta["match_distance_km"]
    for d in meta.get("details") or []:
        if not isinstance(d, dict):
            continue
        key = str(d.get("label", "")).strip().lower().replace(" ", "_")
        if key:
            out[f"meta_{key}"] = d.get("value")
    return out
