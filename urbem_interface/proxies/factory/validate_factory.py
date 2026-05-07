"""
Pre-flight checks for proxy factory job: resolved paths exist before a long build.
"""

from __future__ import annotations

from pathlib import Path

from urbem_interface.proxies.factory.config_loader import load_job


def validate_proxy_factory_config(config_path: Path) -> dict:
    """
    Load factory JSON and verify input paths on disk.

    Returns:
      ok: bool
      error: optional load error message
      missing: list of {key, label, path}
      path_base, output_dir: resolved strings when load succeeded
    """
    path = Path(config_path).resolve()
    try:
        job = load_job(path)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "missing": [],
            "path_base": None,
            "output_dir": None,
        }

    labels = {
        "corine_raster": "CORINE reference raster",
        "corine_gdb": "CORINE File Geodatabase",
        "eprtr_gpkg": "E-PRTR GeoPackage",
        "shipping_routes_shp": "Shipping routes shapefile",
        "population_raster": "Population raster",
    }

    missing: list[dict[str, str]] = []

    if not job.path_base.is_dir():
        missing.append(
            {
                "key": "path_base",
                "label": "Path base (config root)",
                "path": str(job.path_base),
            }
        )

    for key, p in job.paths.items():
        if not p.exists():
            missing.append(
                {
                    "key": key,
                    "label": labels.get(key, key),
                    "path": str(p),
                }
            )
            continue
        if key == "corine_gdb":
            if not p.is_dir():
                missing.append(
                    {
                        "key": key,
                        "label": labels.get(key, key) + " (not a directory)",
                        "path": str(p),
                    }
                )
        elif not p.is_file():
            missing.append(
                {
                    "key": key,
                    "label": labels.get(key, key) + " (not a file)",
                    "path": str(p),
                }
            )

    return {
        "ok": len(missing) == 0,
        "error": None,
        "missing": missing,
        "path_base": str(job.path_base),
        "output_dir": str(job.output_dir),
    }
