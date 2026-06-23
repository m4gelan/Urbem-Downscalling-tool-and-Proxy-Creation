from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_LABELS = {
    "riurbans": "RI-URBANS",
    "jrc": "JRC",
    "eprtr": "E-PRTR",
    "osm": "Airport polygons (OSM)",
    "corine": "Airport polygons (CORINE)",
    "uwwtd": "UWWTD",
}


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


def sidecar_pollutant_mass(pols: dict, pollutant: str) -> float:
    from UrbEm_Visualizer.pollutants import pollutant_key

    want = pollutant_key(pollutant)
    for key, val in pols.items():
        try:
            if pollutant_key(str(key)) == want:
                return float(val or 0)
        except ValueError:
            continue
    return 0.0


def facility_links(rec: dict[str, Any]) -> list[dict[str, Any]]:
    links = rec.get("facility_links")
    if isinstance(links, list) and links:
        return links
    if rec.get("matched") == "yes":
        flon, flat = rec.get("facility_lon"), rec.get("facility_lat")
        if flon is not None and flat is not None:
            return [{
                "facility_id": rec.get("facility_id"),
                "facility_lon": flon,
                "facility_lat": flat,
                "attributed_pollutants": rec.get("pollutants") or {},
                "match_distance_km": rec.get("match_distance_km"),
            }]
    return []


def appointed_meta(rec: dict[str, Any], link: dict[str, Any] | None = None) -> dict[str, Any]:
    src = str(rec.get("match_source") or rec.get("dataset_key") or "").strip().lower()
    if link:
        fid = link.get("facility_id")
        dist = link.get("match_distance_km")
    else:
        fid = rec.get("facility_id")
        dist = rec.get("match_distance_km")
    dataset = rec.get("dataset") or DATASET_LABELS.get(src, src or None)
    details: list[dict[str, str]] = []
    if fid:
        details.append({"label": "Facility id", "value": str(fid)})
    flags = rec.get("flags") or []
    if flags:
        details.append({"label": "Flags", "value": ", ".join(sorted(set(flags)))})
    for d in rec.get("details") or []:
        if isinstance(d, dict) and d.get("label"):
            details.append({"label": str(d["label"]), "value": str(d.get("value", ""))})
    n_links = len(facility_links(rec))
    return {
        "dataset": dataset,
        "dataset_key": src or None,
        "facility_id": str(fid) if fid else None,
        "facility_name": rec.get("facility_name"),
        "match_distance_km": dist,
        "details": details,
        "n_facility_links": n_links if n_links > 1 else None,
    }


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
