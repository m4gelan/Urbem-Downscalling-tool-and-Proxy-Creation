from __future__ import annotations

from UrbEm_Visualizer.dataset_loaders.check import load_expected


def _role_status(sector_cfg: dict, role: str, sector_data: dict | None) -> str:
    mode = sector_cfg["mode"]
    if role == "point_source" and mode not in ("both", "point_only"):
        return "n/a"
    if role == "area_weights" and mode not in ("both", "area_only"):
        return "n/a"
    if not sector_data:
        return "missing"
    entry = sector_data.get(role)
    if not entry:
        return "missing"
    if entry.get("absent"):
        return "absent"
    if entry.get("path"):
        return "ok"
    return "missing"


def build_inputs_table(config: dict) -> list[dict]:
    spec = load_expected()
    sectors_cfg = dict(spec["sectors"])
    optional = spec.get("optional_sectors") or {}
    rows: list[dict] = []

    all_ids = list(sectors_cfg.keys()) + [s for s in optional if s not in sectors_cfg]
    config_sectors = config.get("sectors") or {}

    for sid in all_ids:
        sec_def = sectors_cfg.get(sid) or optional[sid]
        is_opt = sid in optional
        data = config_sectors.get(sid)
        pt = _role_status(sec_def, "point_source", data)
        ar = _role_status(sec_def, "area_weights", data)
        if is_opt and pt == "missing" and ar == "missing":
            if sid not in config_sectors:
                pt = ar = "optional omitted"
        rows.append({
            "sector": sid,
            "optional": is_opt,
            "mode": sec_def["mode"],
            "point_source": pt,
            "area_weights": ar,
        })
    return rows
