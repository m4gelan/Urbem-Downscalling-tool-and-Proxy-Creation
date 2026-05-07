"""
Download Copernicus / WEkEO HRL Imperviousness Density (100m) via HDA.

Country scope: bounding box from NUTS boundaries (``CNTR_CODE`` + dissolve), same idea as
``INPUT/Preprocess/OSM/create_country_specific_packages.py``.

Authentication (same as legacy script):
  - ``HDA_USER`` / ``HDA_PASSWORD``, or
  - ``~/.hdarc`` with ``user:`` / ``password:``

Requires: ``hda``, ``requests``; optional merge: ``rasterio``.

Note: ``hda.Client.search()`` paginates internally — do not loop ``startIndex`` yourself.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bbox_wgs84_from_nuts(nuts_path: Path, cntr_code: str) -> tuple[float, float, float, float]:
    import geopandas as gpd

    gdf = gpd.read_file(nuts_path)
    if "CNTR_CODE" not in gdf.columns:
        raise SystemExit(f"{nuts_path}: missing CNTR_CODE column")
    if "LEVL_CODE" in gdf.columns:
        n0 = gdf[gdf["LEVL_CODE"] == 0].copy()
        if not n0.empty:
            gdf = n0
    cc = cntr_code.strip().upper()
    gdf = gdf[gdf["CNTR_CODE"].astype(str).str.upper().str.strip() == cc].copy()
    if gdf.empty:
        raise SystemExit(f"No NUTS rows for CNTR_CODE={cc!r} in {nuts_path}")
    union = gdf.dissolve().geometry.iloc[0]
    if union is None or union.is_empty:
        raise SystemExit("Dissolved country geometry is empty")
    b = gpd.GeoDataFrame(geometry=[union], crs=gdf.crs).to_crs(4326).total_bounds
    west, south, east, north = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return west, south, east, north


def build_query(
    bbox: list[float] | None,
    *,
    product_type: str | None,
    resolution: str | None,
    year: int | None,
    items_per_page: int,
) -> dict[str, Any]:
    q: dict[str, Any] = {
        "dataset_id": "EO:EEA:DAT:HRL:IMP",
        "itemsPerPage": int(items_per_page),
    }
    if bbox is not None:
        q["bbox"] = bbox
    if product_type is not None:
        q["productType"] = product_type
    if resolution is not None:
        q["resolution"] = resolution
    if year is not None:
        q["year"] = int(year)
    return q


def _collect_tifs(download_dir: Path) -> list[Path]:
    import glob as glob_mod

    pattern = str(download_dir / "**" / "*.tif")
    tifs = sorted(glob_mod.glob(pattern, recursive=True))
    tifs += sorted(glob_mod.glob(str(download_dir / "*.tif")))
    tifs += sorted(glob_mod.glob(str(download_dir / "**" / "*.tiff"), recursive=True))
    return sorted({Path(p).resolve() for p in tifs})


def _extract_zips_for_merge(download_dir: Path, *, verbose: bool) -> Path:
    """If products are .zip, unpack GeoTIFFs into _extract_for_merge/."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):  # type: ignore[misc, no-untyped-def]
            return x

    extract_root = download_dir / "_extract_for_merge"
    zips = sorted(download_dir.glob("*.zip")) + sorted(download_dir.glob("**/*.zip"))
    zips = [p for p in zips if "_extract_for_merge" not in str(p)]
    if not zips:
        return extract_root
    extract_root.mkdir(parents=True, exist_ok=True)
    for zpath in tqdm(zips, desc="Extract zips", unit="zip"):
        sub = extract_root / zpath.stem
        sub.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(sub)
        except zipfile.BadZipFile:
            if verbose:
                print(f"[merge] skip bad zip: {zpath}")
    if verbose:
        print(f"[merge] extracted {len(zips)} zip(s) under {extract_root}")
    return extract_root


def merge_tifs_to_mosaic(download_dir: Path, merge_out: str | Path, *, verbose: bool) -> Path:
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge
    except ImportError as e:
        raise RuntimeError("merge needs rasterio: pip install rasterio") from e

    tifs = _collect_tifs(download_dir)
    if not tifs:
        _extract_zips_for_merge(download_dir, verbose=verbose)
        tifs = _collect_tifs(download_dir / "_extract_for_merge")
    if not tifs:
        raise RuntimeError(
            f"No .tif/.tiff under {download_dir} (or inside zips after extract). "
            "Unpack WEkEO zips or place GeoTIFFs in the download folder."
        )
    if verbose:
        print(f"[merge] merging {len(tifs)} raster file(s) -> {merge_out}")

    srcs = [rasterio.open(p) for p in tifs]
    try:
        mosaic, out_transform = rio_merge(srcs)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "count": mosaic.shape[0],
            }
        )
    finally:
        for s in srcs:
            s.close()

    merge_path = Path(merge_out).resolve()
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(merge_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    print(f"[merge] wrote {merge_path}")
    return merge_path


def main() -> int:
    root = _project_root()
    default_nuts = root / "INPUT" / "Proxy" / "Boundaries" / "NUTS_RG_20M_2021_3035.gpkg"

    parser = argparse.ArgumentParser(
        description="Download WEkEO HRL Imperviousness Density (2018, 100m) via HDA for a NUTS country bbox."
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="NUTS CNTR_CODE (e.g. EL, DE, FR). Used with --nuts-gpkg to set the search bbox unless --bbox is set.",
    )
    parser.add_argument(
        "--nuts-gpkg",
        type=Path,
        default=default_nuts,
        help=f"NUTS boundaries (default: {default_nuts})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information (query, result counts, sample ids).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list result ids (no download).",
    )
    parser.add_argument(
        "--include-id-substr",
        action="append",
        default=[],
        help="Keep only results whose id contains this substring. Can be specified multiple times.",
    )
    parser.add_argument(
        "--exclude-id-substr",
        action="append",
        default=[],
        help="Drop results whose id contains this substring. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-download",
        type=int,
        default=None,
        help="Maximum number of results to download (after filtering).",
    )
    parser.add_argument(
        "--dataset-info",
        action="store_true",
        help="Print dataset metadata for EO:EEA:DAT:HRL:IMP and exit.",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Omit bbox filter to test if any results exist at all.",
    )
    parser.add_argument(
        "--use-filters",
        action="store_true",
        help="Include productType/resolution/year in the HDA query (optional).",
    )
    parser.add_argument(
        "--all-resolutions",
        action="store_true",
        help="Do not restrict to 100 m tiles. Default is 100 m only.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Override NUTS: explicit WGS84 bbox (min_lon min_lat max_lon max_lat).",
    )
    parser.add_argument(
        "--product-type",
        type=str,
        default="Imperviousness Density",
        help="HDA productType when --use-filters is set.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="100m",
        help="HDA resolution when --use-filters is set.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2018,
        help="HDA year when --use-filters is set.",
    )
    parser.add_argument(
        "--items-per-page",
        type=int,
        default=200,
        help="HDA query itemsPerPage parameter.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent download workers used by HDA.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Per-request HTTP timeout for HDA (prevents silent hangs).",
    )
    parser.add_argument(
        "--retry-max",
        type=int,
        default=3,
        help="Max retries for failed HDA requests.",
    )
    parser.add_argument(
        "--sleep-max",
        type=int,
        default=3,
        help="Max backoff sleep between retries.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (debug only).",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Connectivity check to WEkEO gateway (no HDA search).",
    )
    parser.add_argument(
        "--auth-check",
        action="store_true",
        help="Check authentication: token + list 1 dataset, then exit.",
    )
    parser.add_argument(
        "--no-accept-tac",
        action="store_true",
        help="Do not auto-accept Terms & Conditions for EO:EEA:DAT:HRL:IMP.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=None,
        help="Pass limit=N to client.search(query, limit=N) for faster debugging (e.g. 1).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory for downloaded files. Default: INPUT/Preprocess/Imperviousness/_source/wekeo_hrl_imp/<CC>/.",
    )
    parser.add_argument(
        "--merge-out",
        type=str,
        default=None,
        help="Merge all .tif under download-dir into this GeoTIFF (needs rasterio).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip WEkEO search/download; only extract zips + merge --merge-out (offline).",
    )
    args = parser.parse_args()

    cc: str | None = None
    if not args.no_download:
        if args.bbox is not None:
            bbox_list = [float(x) for x in args.bbox]
        elif args.no_bbox:
            bbox_list = None  # type: ignore[assignment]
        else:
            if not args.country:
                raise SystemExit("Set --country (NUTS code) for the search bbox, or pass explicit --bbox, or use --no-bbox.")
            cc = args.country.strip().upper()
            nuts = args.nuts_gpkg.expanduser().resolve()
            if not nuts.is_file():
                raise SystemExit(f"NUTS file not found: {nuts}")
            w, s, e, n = _bbox_wgs84_from_nuts(nuts, cc)
            bbox_list = [w, s, e, n]
            if args.verbose:
                print(f"[bbox] {cc} from NUTS: west,south,east,north = {bbox_list}")
    else:
        bbox_list = [0.0, 0.0, 0.0, 0.0]  # unused

    if not args.all_resolutions:
        include_extra = ["_R100m_"] + [s for s in args.include_id_substr if s]
    else:
        include_extra = list(args.include_id_substr)

    if args.download_dir is not None:
        download_dir = Path(args.download_dir).expanduser().resolve()
    else:
        if cc is None and args.country:
            cc = args.country.strip().upper()
        sub = cc or "custom_bbox"
        download_dir = (root / "INPUT" / "Preprocess" / "Imperviousness" / "_source" / "wekeo_hrl_imp" / sub).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        if not args.merge_out:
            raise SystemExit("--no-download requires --merge-out PATH.tif")
        merge_tifs_to_mosaic(download_dir, args.merge_out, verbose=bool(args.verbose))
        return 0

    if not (os.getenv("HDA_USER") and os.getenv("HDA_PASSWORD")):
        hdarc = Path.home() / ".hdarc"
        if not hdarc.exists():
            raise RuntimeError(
                "Missing WEkEO HDA credentials. Set HDA_USER and HDA_PASSWORD, "
                "or create ~/.hdarc with user:/password: lines."
            )

    import requests
    from hda import Client
    from hda.api import BROKER_URL, Configuration

    verify_ssl = not bool(args.insecure)
    conf = Configuration(verify=verify_ssl)
    client = Client(
        config=conf,
        max_workers=int(args.max_workers),
        timeout=int(args.timeout_seconds),
        retry_max=int(args.retry_max),
        sleep_max=int(args.sleep_max),
    )

    if args.verbose:
        proxy_vars = {k: os.environ.get(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]}
        print(f"[hda] broker_url={BROKER_URL} verify_ssl={verify_ssl}")
        print("[hda] proxies=" + json.dumps(proxy_vars, indent=2))

    if args.ping:
        t0 = time.time()
        try:
            r = requests.get(BROKER_URL, timeout=int(args.timeout_seconds), verify=verify_ssl)
            dt = time.time() - t0
            print(f"[ping] GET {BROKER_URL} -> {r.status_code} in {dt:.2f}s")
            print("[ping] first_bytes=" + repr(r.text[:200]))
        except Exception as e:
            dt = time.time() - t0
            raise RuntimeError(f"[ping] GET {BROKER_URL} failed after {dt:.2f}s: {e}") from e
        return 0

    if args.auth_check:
        t0 = time.time()
        try:
            if args.verbose:
                print("[auth] retrieving token...")
            tok = client.token
            dt = time.time() - t0
            tok_preview = str(tok)[:12] + "..." if tok else None
            print(f"[auth] token_retrieved_in={dt:.2f}s token_preview={tok_preview!r}")
        except Exception as e:
            dt = time.time() - t0
            raise RuntimeError(f"[auth] token retrieval failed after {dt:.2f}s: {e}") from e

        t1 = time.time()
        try:
            if args.verbose:
                print("[auth] requesting datasets(limit=1)...")
            ds = client.datasets(1)
            dt = time.time() - t1
            print(f"[auth] datasets_ok_in={dt:.2f}s count={len(ds) if isinstance(ds, list) else 'n/a'}")
            print(json.dumps(ds, indent=2)[:2000])
        except Exception as e:
            dt = time.time() - t1
            raise RuntimeError(f"[auth] datasets() failed after {dt:.2f}s: {e}") from e
        return 0

    if args.dataset_info:
        try:
            info = client.dataset("EO:EEA:DAT:HRL:IMP")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch dataset info for EO:EEA:DAT:HRL:IMP: {e}") from e
        print(json.dumps(info, indent=2))
        return 0

    if not args.no_accept_tac:
        t0 = time.time()
        try:
            if args.verbose:
                print("[tac] accepting terms for EO:EEA:DAT:HRL:IMP ...")
            client.accept_tac("EO:EEA:DAT:HRL:IMP")
            dt = time.time() - t0
            print(f"[tac] accept_tac_ok_in={dt:.2f}s")
        except Exception as e:
            dt = time.time() - t0
            raise RuntimeError(f"[tac] accept_tac failed after {dt:.2f}s: {e}") from e

    bbox: list[float] | None = None if args.no_bbox else bbox_list
    if args.use_filters:
        product_type = None if args.product_type is None else str(args.product_type)
        resolution = None if args.resolution is None else str(args.resolution)
        year = None if args.year is None else int(args.year)
    else:
        product_type = None
        resolution = None
        year = None

    remaining: int | None = int(args.max_download) if args.max_download is not None else None

    query = build_query(
        bbox,
        product_type=product_type,
        resolution=resolution,
        year=year,
        items_per_page=int(args.items_per_page),
    )
    if args.verbose:
        print("[hda] single search (hda paginates internally; do not loop startIndex)")
        print("[hda] query=" + json.dumps(query, indent=2))

    try:
        if args.verbose:
            print("[hda] searching...")
        t0 = time.time()
        matches = client.search(query, limit=args.search_limit)
        if args.verbose:
            print(f"[hda] search_done_in={time.time() - t0:.2f}s")
    except Exception as e:
        raise RuntimeError(f"HDA search failed: {e}") from e

    results = list(getattr(matches, "results", []) or [])
    n = len(results)
    if args.verbose:
        print(f"[hda] total_results={n}")
        try:
            sample = []
            for item in results[:5]:
                if isinstance(item, dict) and "id" in item:
                    sample.append(item["id"])
                else:
                    sample.append(str(item)[:120])
            if sample:
                print("[hda] sample_ids=" + json.dumps(sample, indent=2))
        except Exception:
            pass

    def _keep(item: Any) -> bool:
        if not isinstance(item, dict) or "id" not in item:
            return True
        _id = str(item["id"])
        for s in include_extra:
            if s and s not in _id:
                return False
        for s in args.exclude_id_substr:
            if s and s in _id:
                return False
        return True

    filtered = [r for r in results if _keep(r)]
    if args.verbose and (include_extra or args.exclude_id_substr):
        print(f"[hda] filtered_results={len(filtered)} (from {len(results)})")
    if len(results) > 0 and len(filtered) == 0 and args.verbose:
        print(
            "[hda] hint: id filter removed all results; try --all-resolutions or adjust "
            "--include-id-substr / --exclude-id-substr; or --use-filters with portal values."
        )

    if remaining is not None:
        filtered = filtered[:remaining]

    if args.list_only:
        for item in filtered:
            if isinstance(item, dict) and "id" in item:
                print(item["id"])
            else:
                print(str(item))
    else:
        try:
            from hda.api import SearchResults
        except Exception as e:
            raise RuntimeError(f"Unable to import hda.api.SearchResults: {e}") from e

        if not filtered:
            print("[hda] nothing to download after filtering.")
        else:
            filtered_matches = SearchResults(client, filtered, "EO:EEA:DAT:HRL:IMP")
            try:
                print(f"[hda] downloading {len(filtered)} product(s) -> {download_dir}", flush=True)
                filtered_matches.download(download_dir=str(download_dir))
            except Exception as e:
                raise RuntimeError(f"HDA download failed, download_dir={download_dir}: {e}") from e

    if args.merge_out:
        merge_tifs_to_mosaic(download_dir, args.merge_out, verbose=bool(args.verbose))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
