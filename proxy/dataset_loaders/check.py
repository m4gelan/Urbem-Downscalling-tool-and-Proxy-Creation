from __future__ import annotations

from pathlib import Path

from proxy.core import log
from proxy.core.alias import resolve_osm_filepath
from proxy.dataset_loaders.load_waste_rasters import resolve_imperviousness_filepath

def require_filepaths_exist(
    repo_root: Path,
    filepaths: dict | None,
    sector_config_path: Path,
    country_profile: dict[str, str] | None = None,
) -> None:
    cfg = sector_config_path.resolve()

    def _msg(detail: str) -> str:
        return f"In this configuration file ({cfg}): {detail}"

    if not filepaths:
        raise ValueError(_msg("no 'filepaths' block (or it is empty)"))
    if not isinstance(filepaths, dict):
        raise TypeError(_msg("'filepaths' must be a mapping"))

    for label, spec in filepaths.items():
        disp = str(label).strip()
        rel = None
        folder_rel = None

        if isinstance(spec, str):
            rel = spec.strip()
        elif isinstance(spec, dict):
            rel = (spec.get("path") or "").strip()
            folder_rel = (spec.get("folder") or "").strip()
        else:
            raise TypeError(_msg(f"{disp!r}: expected string or {{path: ...}}"))

        # Not an error if a folder is provided and exists:
        if folder_rel:
            folder_p = (repo_root / folder_rel.replace("\\", "/")).resolve()
            if folder_p.is_dir():
                log.info(f"{disp} (folder) : OK")
                continue
            else:
                log.info(f"{disp} (folder) : MISSING")
                # Still check if path is valid before erroring,
                # but don't error out just for 'missing path' if 'folder' is configured.

        if rel:
            if disp.upper() == "OSM":
                if country_profile is None:
                    raise ValueError(_msg(f"{disp}: needs country_profile to resolve Country in path"))
                rel = resolve_osm_filepath(rel, country_profile)
            elif disp.upper() == "IMPERVIOUSNESS":
                if country_profile is None:
                    raise ValueError(_msg(f"{disp}: needs country_profile to resolve ISO3 in path"))
                rel = resolve_imperviousness_filepath(rel, country_profile)
            p = (repo_root / rel.replace("\\", "/")).resolve()
            if p.is_file():
                log.info(f"{disp} : OK")
                continue
            else:
                log.info(f"{disp} : MISSING")
                # Only error if neither file nor (valid) folder was found
                if not folder_rel:
                    raise FileNotFoundError(_msg(f"{disp}: not a file: {p}"))
        else:
            # If no path is provided AND no valid folder, only error if neither exists
            if not folder_rel:
                raise ValueError(_msg(f"{disp!r}: missing path or folder"))
            else:
                # Folder is configured and handled above; if missing, do not error about missing 'path'
                continue

        if folder_rel:
            # If folder is provided, but doesn't exist, error here (only if we didn't already continue above)
            folder_p = (repo_root / folder_rel.replace("\\", "/")).resolve()
            raise FileNotFoundError(_msg(f"{disp}: not a file: {p} or folder: {folder_rel}"))
