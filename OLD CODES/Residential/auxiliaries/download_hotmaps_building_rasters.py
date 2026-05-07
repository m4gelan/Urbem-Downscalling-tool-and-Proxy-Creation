#!/usr/bin/env python3
"""
Download Hotmaps building density GeoTIFFs from GitLab (master branch archives).

Repos (EU28+ coverage, ~100 m grid in EPSG:3035):

  - heat_res_curr_density
  - heat_nonres_curr_density
  - gfa_res_curr_density
  - gfa_nonres_curr_density

Usage (from project root)::

  python Residential/auxiliaries/download_hotmaps_building_rasters.py
  python Residential/auxiliaries/download_hotmaps_building_rasters.py --out-dir data/Hotmaps

Requires: urllib (stdlib), zipfile (stdlib). Large downloads (~hundreds of MB each).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

# GitLab project path -> archive basename (must match GitLab generated zip root folder)
HOTMAPS_REPOS: tuple[tuple[str, str], ...] = (
    ("hotmaps/buildings/heat/heat_res_curr_density", "heat_res_curr_density"),
    ("hotmaps/buildings/heat/heat_nonres_curr_density", "heat_nonres_curr_density"),
    ("hotmaps/buildings/gfa_res_curr_density", "gfa_res_curr_density"),
    ("hotmaps/buildings/gfa_nonres_curr_density", "gfa_nonres_curr_density"),
)

DEFAULT_ARCHIVE_REF = "master"
USER_AGENT = (
    "PDM_local-hotmaps-fetch/1.0 (+https://gitlab.com/hotmaps; residential downscaling)"
)


def _archive_url(project_path: str, basename: str, ref: str) -> str:
    return f"https://gitlab.com/{project_path}/-/archive/{ref}/{basename}-{ref}.zip"


def _http_get_bytes(url: str, timeout_s: int = 600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _iter_data_tif_members(z: zipfile.ZipFile, prefix: str) -> Iterable[zipfile.ZipInfo]:
    needle = f"{prefix}/data/"
    for info in z.infolist():
        if info.is_dir():
            continue
        if needle in info.filename.replace("\\", "/") and info.filename.lower().endswith(
            ".tif"
        ):
            yield info


def extract_hotmaps_zip(
    zip_bytes: bytes,
    *,
    dest_dir: Path,
    basename: str,
    ref_suffix: str,
) -> Path:
    """
    Extract the main ``data/*.tif`` from the repo archive into ``dest_dir``.

    GitLab expands ``/-/archive/<ref>/<basename>-<ref>.zip`` with top folder
    ``<basename>-<ref>/`` (e.g. ``heat_res_curr_density-master``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{basename}.tif"
    out_path = dest_dir / out_name
    root_prefix = f"{basename}-{ref_suffix}"

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        members = list(_iter_data_tif_members(z, root_prefix))
        if not members:
            raise FileNotFoundError(
                f"No data/*.tif in archive for {basename!r} (expected "
                f"{root_prefix}/data/*.tif)."
            )
        member = members[0]
        if len(members) > 1:
            prefer = f"{basename}.tif"
            for m in members:
                if m.filename.replace("\\", "/").endswith(f"data/{prefer}"):
                    member = m
                    break
        out_path.write_bytes(z.read(member))
    return out_path


def download_all(
    out_dir: Path,
    *,
    ref: str = DEFAULT_ARCHIVE_REF,
    timeout_s: int = 600,
    skip_existing: bool = True,
) -> list[Path]:
    written: list[Path] = []
    manifest: dict[str, object] = {
        "ref": ref,
        "out_dir": str(out_dir),
        "files": [],
    }
    for project_path, basename in HOTMAPS_REPOS:
        dest = out_dir / f"{basename}.tif"
        if skip_existing and dest.is_file():
            written.append(dest)
            manifest["files"].append({"name": basename, "path": str(dest), "skipped": True})
            print(f"skip existing {dest}", file=sys.stderr)
            continue
        url = _archive_url(project_path, basename, ref)
        print(f"GET {url}", file=sys.stderr)
        try:
            zbytes = _http_get_bytes(url, timeout_s=timeout_s)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"Download failed: {url}: {e}") from e
        path = extract_hotmaps_zip(zbytes, dest_dir=out_dir, basename=basename, ref_suffix=ref)
        written.append(path)
        manifest["files"].append({"name": basename, "path": str(path), "skipped": False})
        print(f"wrote {path}", file=sys.stderr)

    man_path = out_dir / "hotmaps_download_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest {man_path}", file=sys.stderr)
    return written


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Download Hotmaps building density GeoTIFFs.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "Hotmaps",
        help="Directory for the four GeoTIFFs (default: <project>/data/Hotmaps)",
    )
    ap.add_argument(
        "--ref",
        default=DEFAULT_ARCHIVE_REF,
        help="Git branch or tag for -/archive/ (default: master)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target .tif already exists",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-archive HTTP timeout in seconds (default: 600)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    download_all(
        out_dir,
        ref=str(args.ref),
        timeout_s=int(args.timeout),
        skip_existing=not bool(args.force),
    )


if __name__ == "__main__":
    main()
