"""
Validate proxies folder against proxies.json - check that all referenced files exist.
"""

from pathlib import Path


def validate_proxies_folder(
    proxies_config: dict,
    proxies_folder: Path,
) -> dict:
    """
    Validate that all files referenced in proxies_config exist in proxies_folder.

    Returns:
        available: list of (name, filename) for files that exist
        missing: list of (name, filename) for files that are missing
        extra: list of .tif filenames in folder not referenced in config (informational)
    """
    available = []
    missing = []
    referenced = set()

    for name, filename in proxies_config.get("proxies", {}).items():
        referenced.add(filename)
        path = proxies_folder / filename
        if path.exists():
            available.append((name, filename))
        else:
            missing.append((name, filename))

    ghsl_file = proxies_config.get("ghsl_urbancentre")
    if ghsl_file:
        referenced.add(ghsl_file)
        path = proxies_folder / ghsl_file
        if path.exists():
            available.append(("ghsl_urbancentre", ghsl_file))
        else:
            missing.append(("ghsl_urbancentre", ghsl_file))

    extra = []
    if proxies_folder.exists():
        for p in proxies_folder.iterdir():
            if p.suffix.lower() in (".tif", ".tiff") and p.name not in referenced:
                extra.append(p.name)

    return {
        "available": available,
        "missing": missing,
        "extra": extra,
        "all_ok": len(missing) == 0,
    }
