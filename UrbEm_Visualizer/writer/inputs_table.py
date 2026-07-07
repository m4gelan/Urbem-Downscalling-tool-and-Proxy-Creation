from __future__ import annotations

from UrbEm_Visualizer.dataset_loaders.check import check_input, load_expected


def _role_status_from_check(
    sector_id: str,
    role: str,
    sec_def: dict,
    check: dict,
    config_sectors: dict,
    is_opt: bool,
) -> str:
    mode = sec_def["mode"]
    if role == "point_source" and mode not in ("both", "point_only"):
        return "n/a"
    if role == "area_weights" and mode not in ("both", "area_only"):
        return "n/a"

    for item in check.get("accepted_absent") or []:
        if item.get("sector") == sector_id and item.get("role") == role:
            return "absent"

    for item in check.get("ok_items") or []:
        if item.get("sector") != sector_id:
            continue
        if role == "area_weights" and item.get("roads_categories"):
            return "ok"
        if item.get("kind") == role:
            return "ok"

    sector_data = config_sectors.get(sector_id)
    if sector_data:
        entry = sector_data.get(role)
        if entry:
            if entry.get("absent"):
                return "absent"
            if role == "area_weights" and entry.get("categories") and all(entry["categories"].values()):
                return "ok"
            if entry.get("path"):
                return "ok"

    if is_opt and sector_id not in config_sectors:
        return "optional omitted"
    return "missing"


def build_inputs_table(config: dict) -> list[dict]:
    spec = load_expected()
    sectors_cfg = dict(spec["sectors"])
    optional = spec.get("optional_sectors") or {}
    config_sectors = config.get("sectors") or {}

    country = config.get("country")
    emis_year = config.get("emissions_year")
    if emis_year is None and (config.get("paths") or {}).get("cams"):
        emis_year = infer_emissions_year_from_path(config["paths"]["cams"])
    check = (
        check_input(
            country,
            absent_sources=config.get("absent_sources"),
            emissions_year=emis_year,
        )
        if country
        else {"ok_items": [], "accepted_absent": [], "missing": []}
    )

    all_ids = sorted(set(sectors_cfg) | set(optional), key=str.lower)
    rows: list[dict] = []

    for sid in all_ids:
        sec_def = sectors_cfg.get(sid) or optional[sid]
        is_opt = sid in optional
        pt = _role_status_from_check(sid, "point_source", sec_def, check, config_sectors, is_opt)
        ar = _role_status_from_check(sid, "area_weights", sec_def, check, config_sectors, is_opt)
        rows.append({
            "sector": sid,
            "optional": is_opt,
            "mode": sec_def["mode"],
            "point_source": pt,
            "area_weights": ar,
        })
    return rows
