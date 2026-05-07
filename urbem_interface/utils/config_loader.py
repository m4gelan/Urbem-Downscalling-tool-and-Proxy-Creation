"""
Config loaders and path resolution. Wraps urbem_v3 for consistency.
"""

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_config(path: str | Path) -> dict:
    return load_json(path)


def load_proxies_config(path: str | Path, *, proxies_folder: Path | None = None) -> dict:
    """
    Load downscaling proxy config. If the file is a declarative proxy pipeline
    (pipeline_schema >= 1 or proxy_definitions + downscaling), materialize it.
    When ``proxies_folder`` is set, apply output_file fallbacks for missing Proxy_*.tif.
    """
    from urbem_interface.pipeline.job_config import materialize_proxies_config_file

    return materialize_proxies_config_file(Path(path), proxies_folder=proxies_folder)


def load_snap_mapping(path: str | Path) -> dict:
    return load_json(path)


def load_pointsources_config(path: str | Path) -> dict:
    return load_json(path)


def load_linesources_config(path: str | Path) -> dict:
    return load_json(path)


def _resolve_under_root(rel_or_abs: str | Path, root: Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def resolve_paths(run_config: dict, config_dir: Path) -> dict:
    """
    Resolve relative paths against the repo root (parent of urbem_interface).

    Legacy: paths.data_root + paths.cams_folder / paths.proxies_folder (segment under data_root).
    New: paths.input_root for inputs; paths.output_root + paths.proxy_country for proxies;
    paths.output_root + paths.emission_region for emissions (or top-level output_folder).
    """
    config_dir = Path(config_dir)
    project_root = config_dir.parent.parent
    raw = dict(run_config.get("paths") or {})

    input_key = raw.get("input_root") or raw.get("data_root")
    if input_key is None:
        input_key = "data"
    input_base = _resolve_under_root(input_key, project_root)

    out_raw = raw.get("output_root")
    output_root = _resolve_under_root(out_raw, project_root) if out_raw else None

    paths = dict(raw)
    paths["data_root"] = input_base
    paths["input_root"] = input_base
    if "cams_folder" not in raw:
        raise KeyError('run_config["paths"] must include "cams_folder"')
    paths["cams_folder"] = (input_base / raw["cams_folder"]).resolve()

    proxy_country = raw.get("proxy_country")
    proxies_seg = raw.get("proxies_folder")
    if output_root is not None and proxy_country is not None and proxies_seg is None:
        paths["proxies_folder"] = (output_root / "proxy" / proxy_country).resolve()
    elif proxies_seg is not None:
        if output_root is not None:
            paths["proxies_folder"] = _resolve_under_root(proxies_seg, project_root)
        else:
            paths["proxies_folder"] = (input_base / proxies_seg).resolve()
    else:
        raise KeyError(
            'Need paths.proxies_folder, or paths.output_root + paths.proxy_country for Output/proxy/<country>'
        )

    if run_config.get("output_folder") is not None:
        paths["emission_output_folder"] = _resolve_under_root(
            run_config["output_folder"], project_root
        )
    elif output_root is not None and raw.get("emission_region"):
        paths["emission_output_folder"] = (
            output_root / "emission" / raw["emission_region"]
        ).resolve()
    else:
        raise KeyError(
            'Need top-level output_folder, or paths.output_root + paths.emission_region for Output/emission/<region>'
        )

    return paths
