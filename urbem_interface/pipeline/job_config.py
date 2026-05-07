"""
Declarative proxy pipeline JSON: data inputs, per-proxy build metadata, downscaling
allocation, and SNAP map for future roles.

load_proxies_config() delegates here when the file contains pipeline_schema >= 1
or a top-level "extends" indirection.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

_PIPELINE_SCHEMA_KEY = "pipeline_schema"
_MIN_SCHEMA = 1


def load_raw_pipeline(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    ext = raw.get("extends")
    if isinstance(ext, str) and ext.strip():
        parent = (path.parent / ext).resolve()
        if not parent.is_file():
            raise FileNotFoundError(f"extends target not found: {parent}")
        base = load_raw_pipeline(parent)
        ovr = {k: v for k, v in raw.items() if k != "extends"}
        deep_merge(base, ovr)
        return base
    return raw


def deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def materialize_downscaling_config(
    pipeline: dict[str, Any],
    *,
    proxies_folder: Path | None = None,
) -> dict[str, Any]:
    """
    Build the dict expected by area_sources / line_sources (proxies, gnfr_to_proxy,
    ghsl, uc_*, snap_proxy_map, plus catalog extras).
    """
    defs = pipeline.get("proxy_definitions") or {}
    if not isinstance(defs, dict):
        raise ValueError("proxy_pipeline: missing or invalid proxy_definitions")

    down = pipeline.get("downscaling") or {}
    gnfr_map = down.get("gnfr_to_proxy") or {}
    if not gnfr_map:
        raise ValueError("proxy_pipeline: downscaling.gnfr_to_proxy is required")

    needed_roles = {str(v) for v in gnfr_map.values()}
    proxies: dict[str, str] = {}
    fallback_map: dict[str, list[str]] = {}

    for rid in sorted(needed_roles):
        spec = defs.get(rid)
        if not isinstance(spec, dict):
            raise ValueError(
                f"proxy_pipeline: gnfr_to_proxy references unknown proxy id {rid!r}"
            )
        out = spec.get("output_file")
        if not out:
            raise ValueError(f"proxy_pipeline: proxy_definitions[{rid!r}] needs output_file")
        proxies[rid] = str(out)
        fbs = spec.get("fallback_files")
        if isinstance(fbs, list) and fbs:
            fallback_map[rid] = [str(x) for x in fbs]

    out_cfg: dict[str, Any] = {
        "proxies": proxies,
        "gnfr_to_proxy": {str(k): str(v) for k, v in gnfr_map.items()},
        "ghsl_urbancentre": down.get("ghsl_urbancentre", "ghs_europe_iso3.tif"),
        "uc_apply_to_gnfr": down.get("uc_apply_to_gnfr", []),
        "uc_factor": float(down.get("uc_factor", 3)),
        "snap_proxy_map": {
            str(k): str(v) for k, v in (down.get("snap_proxy_map") or {}).items()
        },
        "pipeline_schema": pipeline.get(_PIPELINE_SCHEMA_KEY),
    }

    aux = pipeline.get("auxiliary_proxies")
    if aux:
        out_cfg["auxiliary_proxies"] = copy.deepcopy(aux)
    sem = pipeline.get("semantic_proxy_roles")
    if sem:
        out_cfg["semantic_proxy_roles"] = copy.deepcopy(sem)
    fd = pipeline.get("future_disaggregation")
    if fd:
        out_cfg["future_disaggregation"] = copy.deepcopy(fd)

    if proxies_folder is not None:
        _apply_fallback_files(out_cfg, fallback_map, Path(proxies_folder))

    merged = copy.deepcopy(out_cfg)
    from urbem_interface.pipeline.catalog import merge_proxy_catalog

    merged = merge_proxy_catalog(merged)
    return merged


def _apply_fallback_files(
    cfg: dict[str, Any],
    fallback_map: dict[str, list[str]],
    folder: Path,
) -> None:
    """If primary output_file is missing, use first existing fallback (migration)."""
    proxies = cfg["proxies"]
    for pid, primary in list(proxies.items()):
        primary_path = folder / primary
        if primary_path.is_file():
            continue
        for alt in fallback_map.get(pid, []):
            if (folder / alt).is_file():
                proxies[pid] = alt
                break


def is_pipeline_document(raw: dict[str, Any]) -> bool:
    if raw.get(_PIPELINE_SCHEMA_KEY) is not None:
        try:
            return int(raw[_PIPELINE_SCHEMA_KEY]) >= _MIN_SCHEMA
        except (TypeError, ValueError):
            return False
    return "proxy_definitions" in raw and "downscaling" in raw


def materialize_proxies_config_file(
    path: str | Path,
    *,
    proxies_folder: Path | None = None,
) -> dict[str, Any]:
    p = Path(path).resolve()
    with open(p, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    if raw.get("extends"):
        raw = load_raw_pipeline(p)
    if is_pipeline_document(raw):
        return materialize_downscaling_config(raw, proxies_folder=proxies_folder)
    return raw


def get_factory_inputs(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Subset of pipeline for proxies.factory / UI (paths, crs, bundled json paths)."""
    return copy.deepcopy(pipeline.get("inputs") or {})


def get_proxy_definitions(pipeline: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(pipeline.get("proxy_definitions") or {})
