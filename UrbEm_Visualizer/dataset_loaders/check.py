from __future__ import annotations

import re
from pathlib import Path

import yaml

from UrbEm_Visualizer.paths import config_dir, project_root

_EXPECTED_NAME = "expected_inputs_filepaths.yaml"
_TIF_RE = re.compile(
    r"^(.+)_([A-Za-z_]+)_(?:point_source|area_weights)_(\d{4})\.tif$"
)
_F_ROADS_TIF_RE = re.compile(r"^F_Roads_([A-Za-z_]+)_(F\d+)_(\d{4})\.tif$")


def load_expected() -> dict:
    path = config_dir() / _EXPECTED_NAME
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    for key in ("input_root", "cams", "proxy_weights_root", "sectors"):
        if key not in data:
            raise KeyError(f"{path}: missing required key {key!r}")
    cams = data["cams"]
    if "files" not in cams or "default_year" not in cams:
        raise KeyError(f"{path}: cams must define files and default_year")
    return data


def list_emissions_years(spec: dict | None = None) -> list[int]:
    spec = spec or load_expected()
    return sorted(int(y) for y in spec["cams"]["files"])


def cams_path_for_emissions_year(emissions_year: int, spec: dict | None = None) -> str:
    spec = spec or load_expected()
    files = spec["cams"]["files"]
    key = int(emissions_year)
    if key in files:
        return str(files[key])
    skey = str(key)
    if skey in files:
        return str(files[skey])
    raise KeyError(f"no CAMS file configured for emissions year {key}")


def default_emissions_year(spec: dict | None = None) -> int:
    spec = spec or load_expected()
    return int(spec["cams"]["default_year"])


def infer_emissions_year_from_path(path_str: str) -> int | None:
    m = re.search(r"emissions_year(\d{4})", str(path_str), re.I)
    return int(m.group(1)) if m else None


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


def _roads_category_names() -> list[str]:
    try:
        from UrbEm_Visualizer.downscaling.sector_meta import roads_category_names
        return roads_category_names()
    except Exception:
        return ["F1", "F2", "F3", "F4"]


def _country_tag(country: str) -> str:
    return str(country).strip().replace(" ", "_")


def _country_tags(country: str) -> list[str]:
    tags: list[str] = []

    def add(val: str) -> None:
        t = str(val).strip().replace(" ", "_")
        if t and t not in tags:
            tags.append(t)

    add(country)
    try:
        from proxy.core.alias import resolve_country_profile

        prof = resolve_country_profile(country)
        add(prof["full_name"])
        add(prof["ISO3"])
        add(prof["Abbreviation"])
        add(prof.get("other", ""))
    except Exception:
        pass
    return tags


def _find_roads_proxy(folder: Path, cat: str, year: int, country: str) -> Path | None:
    tag_set = {t.casefold() for t in _country_tags(country)}
    for tag in _country_tags(country):
        p = folder / f"F_Roads_{tag}_{cat}_{year}.tif"
        if p.is_file():
            return p
    hits = sorted(folder.glob(f"F_Roads_*_{cat}_{year}.tif"))
    for p in hits:
        m = _F_ROADS_TIF_RE.match(p.name)
        if m and m.group(1).casefold() in tag_set:
            return p
    if len(hits) == 1:
        return hits[0]
    return None


def _all_sector_defs(spec: dict) -> list[tuple[str, dict, bool]]:
    """(sector_id, sector_cfg, is_optional)."""
    out = [(sid, sec, False) for sid, sec in spec["sectors"].items()]
    for sid, sec in (spec.get("optional_sectors") or {}).items():
        out.append((sid, sec, True))
    return out


def list_countries(root: Path | None = None, emissions_year: int | None = None) -> list[str]:
    root = root or project_root()
    spec = load_expected()
    pw = _resolve(root, spec["proxy_weights_root"])
    year = int(emissions_year) if emissions_year is not None else default_emissions_year(spec)
    found: set[str] = set()
    tag_re = re.compile(r"^(.+)_([A-Za-z_]+)_(?:point_source|area_weights)_(\d{4})\.tif$")
    for sector_id, _, _ in _all_sector_defs(spec):
        folder = pw / sector_id
        if not folder.is_dir():
            continue
        for p in folder.glob("*.tif"):
            m = tag_re.match(p.name)
            if m and m.group(1) == sector_id and int(m.group(3)) == year:
                found.add(m.group(2).replace("_", " "))
                continue
            rm = _F_ROADS_TIF_RE.match(p.name)
            if rm and int(rm.group(3)) == year:
                found.add(rm.group(1).replace("_", " "))
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
    tag = _country_tag(country)
    prompt = sec.get("missing_prompt")
    if is_optional and not prompt:
        prompt = (
            f"No file found for {sector_id}. This sector is optional for some countries — "
            "do you accept? Otherwise, we advise checking if a proxy file can be created "
            "in the proxy pipeline."
        )

    if sector_id == "F_Roads" and mode in ("both", "area_only"):
        cats = _roads_category_names()
        cat_paths: dict[str, str] = {}
        missing_cats: list[str] = []
        for cat in cats:
            fp = _find_roads_proxy(folder, cat, year, country)
            fname = f"F_Roads_{tag}_{cat}_{year}.tif"
            if fp is not None:
                cat_paths[cat] = str(fp)
            else:
                missing_cats.append(fname)
        key = (sector_id, "area_weights")
        if len(cat_paths) == len(cats):
            ok_items.append({
                "kind": "area_weights",
                "sector": sector_id,
                "roads_categories": cat_paths,
                "filename": ", ".join(sorted(cat_paths)),
            })
        elif key in waived:
            accepted_absent.append({
                "sector": sector_id,
                "role": "area_weights",
                "filename": ", ".join(missing_cats),
                "optional": is_optional,
            })
        else:
            for fname in missing_cats:
                missing.append({
                    "kind": "area_weights",
                    "sector": sector_id,
                    "path": str(folder / fname),
                    "filename": fname,
                    "waivable": True,
                    "optional": is_optional,
                    "prompt": prompt.strip() if prompt else "",
                })
        return

    names: dict[str, str] = {}
    if mode in ("both", "area_only"):
        names["area_weights"] = f"{sector_id}_{tag}_area_weights_{year}.tif"
    if mode in ("both", "point_only"):
        names["point_source"] = f"{sector_id}_{tag}_point_source_{year}.tif"

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
    emissions_year: int | None = None,
) -> dict:
    root = root or project_root()
    country = str(country).strip()
    if not country:
        raise ValueError("country is required")

    spec = load_expected()
    proxy_year = int(emissions_year) if emissions_year is not None else default_emissions_year(spec)
    emissions_year = proxy_year
    input_root = _resolve(root, spec["input_root"])
    cams_path = _resolve(root, cams_path_for_emissions_year(emissions_year, spec))
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
        wrong = input_root / "emissions" / cams_path.name
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
            year=proxy_year,
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
        "emissions_year": emissions_year,
        "proxy_year": proxy_year,
        "message": msg,
        "missing": missing,
        "ok_items": ok_items,
        "accepted_absent": accepted_absent,
        "input_root": str(input_root),
        "proxy_weights_root": str(pw_root),
        "cams_path": str(cams_path),
    }
