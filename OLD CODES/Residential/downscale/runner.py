from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from .constants import MODEL_CLASSES
from .eurostat_loader import load_f_enduse_for_country
from .gains_emep import (
    build_M_for_country,
    index_gains_files,
    load_emep,
    load_gains_mapping,
)
from .preprocess import load_and_build_fields


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (root / x)


def load_run_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_country_ids(ds: xr.Dataset) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _country_index_1based(ds: xr.Dataset, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise SystemExit(
            f"Country {iso3!r} not in NetCDF country_id ({len(codes)} countries)."
        ) from exc


def _gnfr_to_index(code: str) -> int:
    gnfr = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F1",
        "F2",
        "F3",
        "F4",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
    )
    c = code.strip().upper()
    if c == "F":
        raise SystemExit("Use F1, F2, F3, or F4 (not F alone).")
    return gnfr.index(c) + 1


def _build_domain_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    country_idx: np.ndarray,
    country_1based: int,
    bbox: tuple[float, float, float, float] | None,
) -> np.ndarray:
    m = country_idx == country_1based
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    return m


def _iso3_for_source(i: int, country_idx: np.ndarray, codes: list[str]) -> str:
    ix = int(country_idx[i]) - 1
    if 0 <= ix < len(codes):
        return str(codes[ix]).strip().upper()
    return "UNK"


def _cams_cell_overlaps_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    bbox: tuple[float, float, float, float],
) -> bool:
    bw, bs, be, bn = bbox
    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south
    return not (east < bw or west > be or north < bs or south > bn)


def _emission_for_spec(
    ds: xr.Dataset,
    i: int,
    spec: dict[str, Any],
    co2_mode: str,
) -> float:
    if spec.get("from_co2_mode"):
        ff = float(np.asarray(ds["co2_ff"].values).ravel()[i])
        bf = float(np.asarray(ds["co2_bf"].values).ravel()[i])
        if co2_mode == "fossil_only":
            return ff
        if co2_mode == "bio_only":
            return bf
        return ff + bf
    var = spec["cams_var"]
    return float(np.asarray(ds[str(var)].values).ravel()[i])


def run(
    config_path: Path | None = None,
    *,
    root: Path | None = None,
    show_progress: bool | None = None,
) -> None:
    root = root or _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cfg_path = config_path or (root / "Residential" / "config" / "residential_cams_downscale.config.json")
    cfg = load_run_config(cfg_path)

    from SourceProxies.grid import first_existing_corine, reference_window_profile, resolve_path

    corine_cfg = cfg.get("corine") or {}
    corine_arg = cfg.get("paths", {}).get("corine")
    corine_path = first_existing_corine(root, corine_arg)
    nuts_gpkg = resolve_path(root, cfg["paths"]["nuts_gpkg"])
    pad_m = float(corine_cfg.get("pad_m", 5000.0))
    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_cntr=str(cfg["country"]["nuts_cntr"]),
        pad_m=pad_m,
    )

    paths = cfg["paths"]
    nc = _resolve(root, paths["cams_nc"])
    if not nc.is_file():
        raise FileNotFoundError(f"CAMS NetCDF not found: {nc}")

    iso_for_enduse = str(cfg["country"]["cams_iso3"]).strip().upper()
    f_enduse = load_f_enduse_for_country(root, iso_for_enduse, cfg)

    emep_path = _resolve(root, paths["emep_ef"])
    emep = load_emep(emep_path)
    mapping_path = _resolve(root, paths["gains_mapping"])
    rules, emep_fuel_hints = load_gains_mapping(mapping_path)

    gains_dir = _resolve(root, paths["gains_dir"])
    overrides = (cfg.get("gains") or {}).get("iso3_file_overrides") or {}
    gains_index = index_gains_files(gains_dir, overrides, root)
    year_col = str((cfg.get("gains") or {}).get("year_column", "2020"))

    pollutant_specs: list[dict[str, Any]] = []
    for p in cfg.get("pollutants") or []:
        pollutant_specs.append(dict(p))
    if not pollutant_specs:
        raise SystemExit("config pollutants list is empty")

    pollutant_outputs: list[str] = []
    for p in pollutant_specs:
        pollutant_outputs.append(str(p["output"]))
    co2_mode = str((cfg.get("co2") or {}).get("mode", "sum_ff_bf"))

    if show_progress is None:
        show_progress = bool((cfg.get("run") or {}).get("show_progress", True))
    else:
        show_progress = bool(show_progress)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    fields = load_and_build_fields(root, cfg, ref)
    X = fields["X"]
    H, W, K = X.shape
    assert K == len(MODEL_CLASSES)

    t_ref = ref["transform"]
    crs_s = ref["crs"]
    bbox_wgs = tuple(float(x) for x in ref["domain_bbox_wgs84"])

    out_dir = _resolve(root, cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc)
    try:
        iso_filter = str(cfg["country"]["cams_iso3"]).strip().upper()
        country_1b = _country_index_1based(ds, iso_filter)
        emis_idx = _gnfr_to_index(str((cfg.get("cams") or {}).get("gnfr", "C")))
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        bbox_cfg = (cfg.get("cams") or {}).get("domain_bbox_wgs84")
        bbox_use = tuple(float(x) for x in bbox_cfg) if bbox_cfg else None
        dom = _build_domain_mask(lon, lat, ci, country_1b, bbox_use)
        source_types = (cfg.get("cams") or {}).get("source_types") or ["area"]
        mask = dom & (emis == emis_idx)
        if "area" in source_types and "point" not in source_types:
            mask = mask & (st == 1)
        elif "point" in source_types and "area" not in source_types:
            mask = mask & (st == 2)
        elif "area" in source_types and "point" in source_types:
            mask = mask & ((st == 1) | (st == 2))

        codes = _decode_country_ids(ds)
        idx_cells = np.flatnonzero(mask)
        M_cache: dict[str, np.ndarray] = {}

        run_cfg = cfg.get("run") or {}
        write_proxy = bool(run_cfg.get("write_residential_sourcearea", True))
        write_emissions = bool(run_cfg.get("write_emissions_geotiffs", False))
        write_per_pollutant_w = bool(run_cfg.get("write_weight_geotiffs", False))
        need_weights = write_proxy or write_per_pollutant_w

        acc = np.zeros((len(pollutant_specs), H, W), dtype=np.float64)
        weights_acc = (
            np.zeros((len(pollutant_specs), H, W), dtype=np.float64)
            if need_weights
            else None
        )

        lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        nlon = int(lon_b.shape[0])
        nlat = int(lat_b.shape[0])
        lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
        lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
        if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
            lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
            lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
        lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
        lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)

        import geopandas as gpd

        for i in idx_cells:
            iso = _iso3_for_source(int(i), ci, codes)
            if iso in M_cache or iso == "UNK":
                continue
            gp = gains_index.get(iso)
            M_cache[iso] = build_M_for_country(
                iso,
                gp,
                year_col,
                rules,
                f_enduse,
                emep,
                pollutant_outputs,
                emep_fuel_hints=emep_fuel_hints,
            )

        cell_iter = idx_cells
        if show_progress and tqdm is not None:
            cell_iter = tqdm(
                idx_cells,
                desc="CAMS cells (residential downscale)",
                unit="cell",
                total=int(idx_cells.size),
                file=sys.stderr,
            )

        for i in cell_iter:
            li, ji = int(lon_ii[i]), int(lat_ii[i])
            west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
            south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west
            if not _cams_cell_overlaps_bbox(west, south, east, north, bbox_wgs):
                continue

            poly4326 = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
            g3035 = poly4326.to_crs(crs_s)
            geom = g3035.geometry.iloc[0]
            minx, miny, maxx, maxy = geom.bounds
            try:
                win = from_bounds(minx, miny, maxx, maxy, transform=t_ref).intersection(
                    Window(0, 0, W, H)
                )
            except WindowError:
                continue
            win = win.round_lengths().round_offsets()
            if win.width < 1 or win.height < 1:
                continue
            r0, c0 = int(win.row_off), int(win.col_off)
            h_win, w_win = int(win.height), int(win.width)
            Xw = X[r0 : r0 + h_win, c0 : c0 + w_win, :].reshape(-1, K).astype(np.float64)
            iso = _iso3_for_source(int(i), ci, codes)
            M = M_cache.get(iso)
            if M is None:
                M = np.zeros((len(pollutant_outputs), K), dtype=np.float64)
            U = Xw @ M.T
            n_pix = U.shape[0]
            rr, cc = np.meshgrid(
                np.arange(r0, r0 + h_win, dtype=np.int32),
                np.arange(c0, c0 + w_win, dtype=np.int32),
                indexing="ij",
            )
            flat_r = rr.ravel()
            flat_c = cc.ravel()
            for pi, spec in enumerate(pollutant_specs):
                u_col = U[:, pi]
                ssum = float(np.sum(u_col))
                if ssum <= 0.0 or not np.isfinite(ssum):
                    w_col = np.full(n_pix, 1.0 / max(n_pix, 1), dtype=np.float64)
                else:
                    w_col = u_col / ssum
                E = _emission_for_spec(ds, int(i), spec, co2_mode)
                if not np.isfinite(E):
                    E = 0.0
                contrib = w_col * E
                np.add.at(acc[pi], (flat_r, flat_c), contrib)
                if weights_acc is not None:
                    np.add.at(weights_acc[pi], (flat_r, flat_c), w_col)

    finally:
        ds.close()

    profile_single = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": crs_s,
        "transform": t_ref,
        "compress": "lzw",
    }
    if write_emissions:
        for pi, spec in enumerate(pollutant_specs):
            out_tif = out_dir / f"emissions_{spec['output']}.tif"
            with rasterio.open(out_tif, "w", **profile_single) as dst:
                dst.write(acc[pi].astype(np.float32), 1)
                dst.set_band_description(1, f"kg_yr_{spec['output']}_GNFR_C_downscaled")

    proxy_path_written: str | None = None
    if write_proxy and weights_acc is not None:
        proxy_rel = (paths.get("residential_sourcearea_tif") or "").strip()
        if not proxy_rel:
            proxy_rel = "SourceProxies/outputs/EL/Residential_sourcearea.tif"
        proxy_tif = _resolve(root, proxy_rel)
        proxy_tif.parent.mkdir(parents=True, exist_ok=True)
        n_b = len(pollutant_specs)
        profile_mb = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": n_b,
            "dtype": "float32",
            "crs": crs_s,
            "transform": t_ref,
            "compress": "lzw",
        }
        stack = np.stack([weights_acc[i] for i in range(n_b)], axis=0)
        with rasterio.open(proxy_tif, "w", **profile_mb) as dst:
            for bi, spec in enumerate(pollutant_specs):
                dst.write(stack[bi].astype(np.float32), bi + 1)
                dst.set_band_description(
                    bi + 1,
                    f"weight_share_residential_{str(spec['output']).lower()}",
                )
        try:
            proxy_path_written = str(proxy_tif.relative_to(root))
        except ValueError:
            proxy_path_written = str(proxy_tif)
        sidecar = {
            "builder": "residential_gnfr_c",
            "output_geotiff": proxy_path_written,
            "bands": [
                {"index": i + 1, "pollutant": str(spec["output"])}
                for i, spec in enumerate(pollutant_specs)
            ],
            "crs": str(crs_s),
            "width": W,
            "height": H,
            "domain_bbox_wgs84": list(bbox_wgs),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        proxy_tif.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote multi-band proxy {proxy_tif}", file=sys.stderr)

    if write_per_pollutant_w and weights_acc is not None:
        for pi, spec in enumerate(pollutant_specs):
            wtif = out_dir / f"weights_{spec['output']}.tif"
            with rasterio.open(wtif, "w", **profile_single) as dst:
                dst.write(weights_acc[pi].astype(np.float32), 1)

    emission_paths = (
        [str(out_dir / f"emissions_{p['output']}.tif") for p in pollutant_specs]
        if write_emissions
        else []
    )
    man = {
        "config": str(cfg_path),
        "cams_nc": str(nc),
        "corine_path": str(corine_path),
        "ref": {
            "crs": crs_s,
            "height": H,
            "width": W,
            "domain_bbox_wgs84": list(bbox_wgs),
        },
        "outputs": emission_paths,
        "residential_sourcearea": proxy_path_written,
        "gains_files_indexed": len(gains_index),
    }
    (out_dir / "downscale_manifest.json").write_text(
        json.dumps(man, indent=2),
        encoding="utf-8",
    )
    if write_emissions:
        print(f"Wrote emissions to {out_dir}", file=sys.stderr)
    elif proxy_path_written:
        print(f"Wrote residential proxy only ({proxy_path_written})", file=sys.stderr)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Residential GNFR C CAMS downscaling (Hotmaps+GAINS+EMEP).")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to residential_cams_downscale.config.json",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: parent of Residential package)",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar over CAMS cells",
    )
    args = ap.parse_args()
    run(
        config_path=args.config,
        root=args.root,
        show_progress=False if args.no_progress else None,
    )


if __name__ == "__main__":
    main()
