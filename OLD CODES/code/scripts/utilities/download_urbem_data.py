#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download UrbEm input data as specified in code/UrbEm/README.md.

Saves files under the PDM data/ folder. Run from the PDM project root, or set
DATA_ROOT to the path containing data/CAMS, data/spatial_proxies, data/E_PRTR.

Usage:
    python code/scripts/utilities/download_urbem_data.py [--data-root PATH] [--skip-zenodo] [--dry-run]

Data that require manual download (registration or web form) are listed at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request
from urllib.error import URLError, HTTPError

# Optional: use requests for streaming large files and better redirect handling
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ------------------------------------------------------------------------------
# Configuration: URLs and paths from UrbEm README and Python script
# ------------------------------------------------------------------------------

# Zenodo: UrbEm spatial proxies (CORINE-derived rasters, shipping, etc.)
ZENODO_RECORD_ID = "5508739"
ZENODO_DOI = "10.5281/zenodo.5508739"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_FILE_KEY = "UrbEm_proxies_v1.0.zip"

# E-PRTR: facilities in KMZ (UrbEm script expects EPRTR_facilities_v17.kmz in E_PRTR/kmz/)
# EEA now serves via datahub; legacy KMZ may still be available from industry portal
EPRTR_KMZ_URL = (
    "https://www.eea.europa.eu/data-and-maps/data/"
    "member-states-reporting-art-7-under-the-european-pollutant-release-and-transfer-register-e-prtr-regulation-23/"
    "e-prtr-facilities-kmz-format/eprtr_facilities_v9.kmz"
)
# If v17 is required, download from industry.eea.europa.eu or SDI and rename to EPRTR_facilities_v17.kmz
EPRTR_CSV_ZIP_URL = (
    "https://www.eea.europa.eu/data-and-maps/data/"
    "member-states-reporting-art-7-under-the-european-pollutant-release-and-transfer-register-e-prtr-regulation-23/"
    "european-pollutant-release-and-transfer-register-e-prtr-data-base/eprtr_v9_csv.zip"
)

# GHS Population: JRC GHSL (often requires form / no direct stable URL)
# Product: GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0
GHS_POP_INFO_URL = "https://ghsl.jrc.ec.europa.eu/ghs_pop2019.php"
GHS_POP_DOWNLOAD_PAGE = "https://ghsl.jrc.ec.europa.eu/download.php?ds=pop"

# CAMS-REG: requires registration at ECCAD
ECCAD_URL = "https://eccad.aeris-data.fr"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def get_project_root() -> Path:
    """PDM project root (parent of 'code', 'data', etc.)."""
    script_dir = Path(__file__).resolve().parent
    # code/scripts/utilities -> utilities -> scripts -> code -> project root
    root = script_dir.parent.parent.parent
    if (root / "data").is_dir() and (root / "code").is_dir():
        return root
    # Fallback: current working directory
    cwd = Path.cwd()
    if (cwd / "data").is_dir():
        return cwd
    return script_dir.parent.parent.parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(
    url: str,
    dest: Path,
    dry_run: bool = False,
    description: str = "",
) -> bool:
    """Download a file from url to dest. Returns True on success."""
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return True
    if dry_run:
        print(f"  [dry-run] would download: {url} -> {dest}")
        return True
    print(f"  downloading: {url}")
    try:
        if HAS_REQUESTS:
            req = requests.get(url, stream=True, timeout=120)
            req.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in req.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            req = Request(url, headers={"User-Agent": "PDM-UrbEm-Download/1.0"})
            urlretrieve(req, dest)
        print(f"  -> {dest}")
        return True
    except (URLError, HTTPError, OSError) as e:
        print(f"  [error] {e}")
        return False


def download_zenodo_file(
    record_id: str,
    file_key: str,
    dest: Path,
    dry_run: bool = False,
) -> bool:
    """Resolve Zenodo record, get file URL, and download."""
    direct_url = f"https://zenodo.org/records/{record_id}/files/{file_key}"
    if dry_run:
        print(f"  [dry-run] would download {direct_url} -> {dest}")
        return True
    if download_file(direct_url, dest, dry_run=False):
        return True
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        import json
        if HAS_REQUESTS:
            r = requests.get(api_url, timeout=30)
            r.raise_for_status()
            data = r.json()
        else:
            req = Request(api_url, headers={"User-Agent": "PDM-UrbEm-Download/1.0"})
            tmp_path, _ = urlretrieve(req)
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
        for f in data.get("files", []):
            if f.get("key") == file_key:
                return download_file(f["links"]["self"], dest, dry_run=False)
        print(f"  [error] file '{file_key}' not found in record")
        return False
    except Exception as e:
        print(f"  [error] {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download UrbEm input data (proxies, E-PRTR). CAMS and GHS-POP require manual steps."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing data/ (default: PDM project root)",
    )
    parser.add_argument(
        "--skip-zenodo",
        action="store_true",
        help="Skip Zenodo proxies zip (e.g. if already present)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded",
    )
    args = parser.parse_args()

    root = args.data_root or get_project_root()
    data_root = root / "data"
    proxies_dir = data_root / "spatial_proxies"
    eprtr_dir = data_root / "E_PRTR"
    eprtr_kmz_dir = eprtr_dir / "kmz"
    cams_dir = data_root / "CAMS"

    print("PDM UrbEm data download")
    print("Data root:", data_root)
    if args.dry_run:
        print("(dry-run: no files written)")
    print()

    # --------------------------------------------------------------------------
    # 1. Zenodo: UrbEm spatial proxies (CORINE-derived, shipping, etc.)
    # --------------------------------------------------------------------------
    if not args.skip_zenodo:
        ensure_dir(proxies_dir)
        zip_path = proxies_dir / ZENODO_FILE_KEY
        print(f"1. Zenodo spatial proxies (DOI {ZENODO_DOI})")
        ok = download_zenodo_file(ZENODO_RECORD_ID, ZENODO_FILE_KEY, zip_path, dry_run=args.dry_run)
        if ok and zip_path.exists() and not args.dry_run:
            print("  extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(proxies_dir)
            print("  extracted to", proxies_dir)
        print()
    else:
        print("1. Zenodo spatial proxies: skipped (--skip-zenodo)")
        print()

    # --------------------------------------------------------------------------
    # 2. E-PRTR: KMZ and CSV
    # --------------------------------------------------------------------------
    ensure_dir(eprtr_kmz_dir)
    print("2. E-PRTR data")
    # UrbEm script expects EPRTR_facilities_v17.kmz; EEA may serve v9, we download and optionally rename
    kmz_dest = eprtr_kmz_dir / "EPRTR_facilities_v17.kmz"
    if not kmz_dest.exists():
        # Try legacy v9 URL; user can replace with v17 from industry.eea.europa.eu if needed
        ok = download_file(EPRTR_KMZ_URL, eprtr_kmz_dir / "eprtr_facilities_v9.kmz", args.dry_run)
        if ok and (eprtr_kmz_dir / "eprtr_facilities_v9.kmz").exists() and not kmz_dest.exists():
            # Copy/symlink so UrbEm finds v17 name; if only v9 exists, copy to v17 for compatibility
            try:
                import shutil
                shutil.copy(eprtr_kmz_dir / "eprtr_facilities_v9.kmz", kmz_dest)
                print("  copied eprtr_facilities_v9.kmz -> EPRTR_facilities_v17.kmz (use official v17 if available)")
            except OSError:
                pass
    else:
        print("  [skip] EPRTR KMZ already present:", kmz_dest)

    csv_zip_dest = eprtr_dir / "eprtr_csv.zip"
    if not csv_zip_dest.exists():
        download_file(EPRTR_CSV_ZIP_URL, csv_zip_dest, args.dry_run)
    else:
        print("  [skip] E-PRTR CSV zip already present")
    print()

    # --------------------------------------------------------------------------
    # 3. CAMS-REG and GHS-POP: manual instructions
    # --------------------------------------------------------------------------
    ensure_dir(cams_dir)
    print("3. Manual downloads required")
    print()
    print("  CAMS-REG emission inventory")
    print("    Register and download at:", ECCAD_URL)
    print("    Select CAMS-REG-ANT (regional anthropogenic); download all pollutants/sectors.")
    print("    Place the .nc files in:", cams_dir)
    print()
    print("  GHS Population raster (GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0)")
    print("    Download from:", GHS_POP_INFO_URL)
    print("    Or:", GHS_POP_DOWNLOAD_PAGE)
    print("    Product: 2015, WGS84, 30 arcsec. Save the .tif to:", proxies_dir)
    print()
    print("  Optional (may be inside Zenodo zip): GHS Urban Centre, CORINE GDB, Shipping Routes.")
    print("    If not present after extracting Zenodo, see UrbEm README for links.")
    print()

    # --------------------------------------------------------------------------
    # Point user to UrbEm Input_Data
    # --------------------------------------------------------------------------
    print("UrbEm expects a single 'Input_Data' folder containing:")
    print("  - GHS_POP_E2015_GLOBE_R2019A_4326_30ss_V1_0.tif")
    print("  - CORINE raster and GDB (often from Zenodo zip)")
    print("  - E_PRTR/kmz/EPRTR_facilities_v17.kmz")
    print("  - GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0/ (urban centre)")
    print("  - Shipping_Routes/shipping_routes_wgs.shp")
    print()
    print("You can either:")
    print("  A) Set UrbEm script InFolder to:", proxies_dir, "(and merge CAMS, E_PRTR paths there), or")
    print("  B) Create a symlink/copy 'Input_Data' that aggregates", data_root, "content.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
