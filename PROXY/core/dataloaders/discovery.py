from __future__ import annotations

from pathlib import Path


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


from PROXY.core.dataloaders.config import project_root as find_project_root, load_yaml, resolve_path

def discover_corine(project_root: Path, configured: Path) -> Path:
    # Instead of guessing, just load corine_tif path from paths.yaml and resolve.
    paths_yaml = (project_root / "PROXY" / "config" / "paths.yaml")
    paths = load_yaml(paths_yaml)
    corine_path = paths.get("proxy_common", {}).get("corine_tif")
    if not corine_path:
        raise KeyError(f"corine_tif not set in {paths_yaml}")
    return resolve_path(project_root, corine_path)

def discover_cams_emissions(project_root: Path, configured: Path) -> Path:
    # Always use the path from paths.yaml
    paths_yaml = project_root / "PROXY" / "config" / "paths.yaml"
    paths = load_yaml(paths_yaml)
    cams_path = paths.get("emissions", {}).get("cams_2019_nc")
    if not cams_path:
        raise KeyError(f"cams_2019_nc not set in {paths_yaml}")
    return resolve_path(project_root, cams_path)

