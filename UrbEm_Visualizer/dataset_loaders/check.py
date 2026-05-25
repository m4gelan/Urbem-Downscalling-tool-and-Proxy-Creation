from __future__ import annotations

import re
from pathlib import Path

import yaml

from UrbEm_Visualizer.paths import config_dir, project_root

_EXPECTED_NAME = "expected_inputs_filepaths.yaml"
_TIF_RE = re.compile(
    r"^(.+)_([A-Za-z]+)_(?:point_source|area_weights)_(\d{4})\.tif$"
)


def load_expected() -> dict:
    path = config_dir() / _EXPECTED_NAME
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    for key in ("year", "input_root", "cams", "proxy_weights_root", "sectors"):
        if key not in data:
            raise KeyError(f"{path}: missing required key {key!r}")
    return data


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def _absent_set(absent_sources: list[dict] | None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in absent_sources or []:
        sector = item.get("sector")
        role = item.get("role")
        if sector and role:
            out.add((str(sector), str(role)))
    return out


def _all_sector_defs(spec: dict) -> list[tuple[str, dict, bool]]:
    """(sector_id, sector_cfg, is_optional)."""
    out = [(sid, sec, False) for sid, sec in spec["sectors"].items()]
    for sid, sec in (spec.get("optional_sectors") or {}).items():
        out.append((sid, sec, True))
    return out


def list_countries(root: Path | None = None) -> list[str]:
    root = root or project_root()
    spec = load_expected()
    pw = _resolve(root, spec["proxy_weights_root"])
    year = int(spec["year"])
    found: set[str] = set()
    for sector_id, _, _ in _all_sector_defs(spec):
        folder = pw / sector_id
        if not folder.is_dir():
            continue
        for p in folder.glob("*.tif"):
            m = _TIF_RE.match(p.name)
            if m and m.group(1) == sector_id and int(m.group(3)) == year:
                found.add(m.group(2))
    return sorted(found, key=str.lower)


def _check_sector_files(
    *,
    sector_id: str,
    sec: dict,
    is_optional: bool,
    country: str,
    year: int,
    folder: Path,
    waived: set[tuple[str, str]],
    missing: list[dict],
    ok_items: list[dict],
    accepted_absent: list[dict],
) -> None:
    mode = sec["mode"]
    names: dict[str, str] = {}
    if mode in ("both", "area_only"):
        names["area_weights"] = f"{sector_id}_{country}_area_weights_{year}.tif"
    if mode in ("both", "point_only"):
        names["point_source"] = f"{sector_id}_{country}_point_source_{year}.tif"

    prompt = sec.get("missing_prompt")
    if is_optional and not prompt:
        prompt = (
            f"No file found for {sector_id}. This sector is optional for some countries — "
            "do you accept? Otherwise, we advise checking if a proxy file can be created "
            "in the proxy pipeline."
        )

    for role, fname in names.items():
        fp = folder / fname
        key = (sector_id, role)
        entry = {
            "kind": role,
            "sector": sector_id,
            "path": str(fp),
            "filename": fname,
            "waivable": True,
            "optional": is_optional,
        }
        if is_optional:
            entry["prompt"] = prompt.strip() if prompt else ""
        if fp.is_file():
            ok_items.append(entry)
        elif key in waived:
            accepted_absent.append({
                "sector": sector_id,
                "role": role,
                "filename": fname,
                "optional": is_optional,
            })
        else:
            missing.append(entry)


def check_input(
    country: str,
    root: Path | None = None,
    absent_sources: list[dict] | None = None,
) -> dict:
    root = root or project_root()
    country = str(country).strip()
    if not country:
        raise ValueError("country is required")

    spec = load_expected()
    year = int(spec["year"])
    input_root = _resolve(root, spec["input_root"])
    cams_path = _resolve(root, spec["cams"]["path"])
    pw_root = _resolve(root, spec["proxy_weights_root"])
    waived = _absent_set(absent_sources)

    missing: list[dict] = []
    ok_items: list[dict] = []
    accepted_absent: list[dict] = []

    if not input_root.is_dir():
        return {
            "ok": False,
            "country": country,
            "message": f"INPUT folder not found: {input_root}",
            "missing": [{"kind": "input_root", "path": str(input_root), "waivable": False}],
            "ok_items": [],
            "accepted_absent": [],
        }

    if not cams_path.is_file():
        wrong = input_root / "emissions" / Path(spec["cams"]["path"]).name
        hint = None
        if wrong.is_file():
            hint = "CAMS must live under INPUT/Emissions (capital E), not INPUT/emissions."
        entry = {
            "kind": "cams",
            "sector": None,
            "path": str(cams_path),
            "label": "CAMS NetCDF",
            "waivable": False,
        }
        if hint:
            entry["hint"] = hint
        missing.append(entry)
    else:
        ok_items.append({"kind": "cams", "path": str(cams_path)})

    if not pw_root.is_dir():
        missing.append({
            "kind": "proxy_weights_root",
            "sector": None,
            "path": str(pw_root),
            "waivable": False,
        })
        return {
            "ok": False,
            "country": country,
            "message": f"Proxy weights folder not found: {pw_root}",
            "missing": missing,
            "ok_items": ok_items,
            "accepted_absent": accepted_absent,
        }

    for sector_id, sec, is_optional in _all_sector_defs(spec):
        folder = pw_root / sector_id
        if not folder.is_dir():
            if is_optional:
                continue
            missing.append({
                "kind": "sector_folder",
                "sector": sector_id,
                "path": str(folder),
                "waivable": False,
            })
            continue
        _check_sector_files(
            sector_id=sector_id,
            sec=sec,
            is_optional=is_optional,
            country=country,
            year=year,
            folder=folder,
            waived=waived,
            missing=missing,
            ok_items=ok_items,
            accepted_absent=accepted_absent,
        )

    ok = len(missing) == 0
    if ok:
        msg = "All expected inputs found."
    elif accepted_absent:
        msg = "Some files missing — confirm optional sectors or add files."
    else:
        msg = "Some expected files are missing."
    return {
        "ok": ok,
        "country": country,
        "message": msg,
        "missing": missing,
        "ok_items": ok_items,
        "accepted_absent": accepted_absent,
        "input_root": str(input_root),
        "proxy_weights_root": str(pw_root),
        "cams_path": str(cams_path),
    }
