from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent
# Reference tables (CORINE class map, E-PRTR SNAP rules) live under urbem_interface/config/
_BUNDLED_ROOT = _PKG.parent.parent / "config" / "factory_bundled"
BUNDLED_CONFIG = _BUNDLED_ROOT


@dataclass
class ProxyFactoryJob:
    """Resolved job: paths on disk + JSON blobs + output directory."""

    path_base: Path
    output_dir: Path
    paths: dict[str, Path]
    crs_projected: str
    crs_wgs: str
    crs_corine: str
    corine_classes: dict[str, Any]
    eprtr_snap_assignment: dict[str, Any]
    population_output_basename: str
    keep_intermediate_shp: bool
    vector_subset: str | None = None
    waste_output_tif: str | None = None
    eprtr_snap9_only: bool = False


def _resolve_path(raw: str | Path, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_job(config_path: Path) -> ProxyFactoryJob:
    """
    Load proxy factory JSON. Relative paths are resolved against `path_base`
    (default: directory containing the config file).

    Required keys:
      paths.corine_raster, corine_gdb, eprtr_gpkg, shipping_routes_shp, population_raster
    Optional:
      output_dir (default: created_proxies under path_base)
      path_base
      crs.projected, crs.wgs84, crs.corine (defaults EPSG:3035 / EPSG:4326 / EPSG:3035)
      corine_classes_json, eprtr_snap_assignment_json (default: bundled copies)
      population_output_filename, proxies_interface_json (to read population key)
      keep_intermediate_shp (default false)
    """
    config_path = config_path.resolve()
    raw = _load_json(config_path)

    pb_raw = raw.get("path_base")
    if pb_raw:
        pb = Path(pb_raw)
        path_base = pb.resolve() if pb.is_absolute() else (config_path.parent / pb).resolve()
    else:
        path_base = config_path.parent.resolve()

    paths_in = raw.get("paths")
    if not isinstance(paths_in, dict):
        raise ValueError("Config must contain a 'paths' object")

    required = (
        "corine_raster",
        "corine_gdb",
        "eprtr_gpkg",
        "shipping_routes_shp",
        "population_raster",
    )
    missing = [k for k in required if k not in paths_in or not str(paths_in[k]).strip()]
    if missing:
        raise ValueError(f"paths section missing or empty: {missing}")

    paths = {k: _resolve_path(paths_in[k], path_base) for k in required}

    out_raw = raw.get("output_dir", "created_proxies")
    output_dir = _resolve_path(out_raw, path_base)

    crs = raw.get("crs") or {}
    crs_projected = str(crs.get("projected", "EPSG:3035"))
    crs_wgs = str(crs.get("wgs84", "EPSG:4326"))
    crs_corine = str(crs.get("corine", "EPSG:3035"))

    cc_key = raw.get("corine_classes_json")
    if cc_key:
        corine_classes = _load_json(_resolve_path(cc_key, path_base))
    else:
        bundled = BUNDLED_CONFIG / "corine_classes.json"
        if not bundled.is_file():
            raise FileNotFoundError(f"Bundled corine_classes.json missing: {bundled}")
        corine_classes = _load_json(bundled)

    es_key = raw.get("eprtr_snap_assignment_json")
    if es_key:
        eprtr_snap_assignment = _load_json(_resolve_path(es_key, path_base))
    else:
        bundled_e = BUNDLED_CONFIG / "eprtr_snap_assignment.json"
        if not bundled_e.is_file():
            raise FileNotFoundError(f"Bundled eprtr_snap_assignment.json missing: {bundled_e}")
        eprtr_snap_assignment = _load_json(bundled_e)

    pop_name = raw.get("population_output_filename")
    if not pop_name:
        pij = raw.get("proxies_interface_json")
        if pij:
            pj_path = _resolve_path(pij, path_base)
            if pj_path.is_file():
                pj_raw = _load_json(pj_path)
                if pj_raw.get("extends"):
                    from urbem_interface.pipeline.job_config import (
                        load_raw_pipeline,
                        materialize_downscaling_config,
                    )

                    merged = load_raw_pipeline(Path(pj_path).resolve())
                    mat = materialize_downscaling_config(merged, proxies_folder=None)
                    pop_name = (mat.get("proxies") or {}).get("population")
                else:
                    pop_name = (pj_raw.get("proxies") or {}).get("population")
        if not pop_name:
            pop_name = "GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif"

    keep_shp = bool(raw.get("keep_intermediate_shp", False))

    vs = raw.get("vector_subset")
    vector_subset = str(vs).strip().lower() if vs is not None and str(vs).strip() else None

    wo = raw.get("waste_output_tif")
    waste_output_tif = str(wo).strip() if wo is not None and str(wo).strip() else None

    eprtr_snap9_only = bool(raw.get("eprtr_snap9_only", False))

    return ProxyFactoryJob(
        path_base=path_base,
        output_dir=output_dir,
        paths=paths,
        crs_projected=crs_projected,
        crs_wgs=crs_wgs,
        crs_corine=crs_corine,
        corine_classes=corine_classes,
        eprtr_snap_assignment=eprtr_snap_assignment,
        population_output_basename=str(pop_name),
        keep_intermediate_shp=keep_shp,
        vector_subset=vector_subset,
        waste_output_tif=waste_output_tif,
        eprtr_snap9_only=eprtr_snap9_only,
    )
