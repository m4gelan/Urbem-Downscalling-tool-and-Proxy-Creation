#!/usr/bin/env python3
"""
CAMS **area** cells (GNFR A): build **spatial weight masks** on CORINE **grid cells**
(100 m pixel footprints as polygons in WGS84).

**Logic (per CAMS cell):**

1. **CORINE** — *eligibility only*: a pixel can get a positive weight only if its class
   is in ``corine_codes`` (e.g. 121 and/or EEA grid 3). All other pixels get weight 0.
2. **LandScan** — *intensity among eligibles*: on eligible pixels only, the unnormalized
   weight is ``max(P, pop_floor) ** pop_exponent`` where ``P`` is the warped LandScan
   value. With ``pop_exponent = 1`` (default), **larger P ⇒ larger weight** (strictly,
   weakly larger after the floor). Then weights are normalized to sum to 1 in the cell.

LandScan is a **population count** per its native cell, resampled onto CORINE; it is
not a formal persons/m² density, but it plays the role of “more population here ⇒
stronger share of the cell total.”

Export also writes ``*_manifest.json`` (input paths + parameters) and ``*_report.md``
(method description) next to the GeoJSON when using the CLI.

Requires: xarray, rasterio, geopandas, numpy, shapely
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None  # type: ignore

# Greater Athens (WGS84): west, south, east, north — for fast demos.
ATHENS_BBOX_WGS84: tuple[float, float, float, float] = (23.55, 37.85, 23.98, 38.08)


def _cams_helper():
    """Load sibling script without requiring PublicPower to be a package."""
    p = Path(__file__).resolve().parent / "cams_A_publicpower_greece.py"
    spec = importlib.util.spec_from_file_location("_cams_a_publicpower_greece", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cams_area_mask_grc(ds: "xr.Dataset") -> np.ndarray:
    """GNFR A + Greece + area sources (reuses same indices as cams_A_publicpower_greece)."""
    cams = _cams_helper()
    grc_1b = cams._country_index_1based(ds, "GRC")
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == cams.IDX_A_PUBLIC_POWER) & cams._build_domain_mask(
        lon, lat, ci, grc_1b, None
    )
    return base & (st == 1)


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


def build_cams_area_corine_landscan_weights(
    ds: "xr.Dataset",
    base_mask: np.ndarray,
    *,
    corine_path: Path,
    landscan_path: Path,
    corine_codes: tuple[int, ...] = (121, 3),
    corine_band: int = 1,
    pop_exponent: float = 1.0,
    pop_floor: float = 0.0,
    landscan_resampling: Literal["bilinear", "nearest"] = "bilinear",
    fallback_if_no_corine: Literal["pop_in_cell", "skip"] = "pop_in_cell",
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
    show_progress: bool = False,
) -> "gpd.GeoDataFrame":
    """
    One row per CORINE raster **pixel footprint** (polygon) with positive normalized weight.

    Columns:

    - ``weight_raw``: before normalization; 0 if CORINE ineligible, else
      ``max(pop, pop_floor)**pop_exponent`` (larger pop ⇒ larger raw weight by default).
    - ``weight_share``: ``weight_raw / sum(weight_raw)`` in the CAMS cell (sums to 1).
    - ``landscan_pop``: **P** after warp to CORINE grid.
    - ``corine_value``: rounded class; ``in_corine_target`` mirrors eligibility.
    - ``weight_basis``: ``corine121_pop`` | ``pop_full_cell`` | ``uniform_cell``.
    - ``lon_cams_centre``, ``lat_cams_centre``, ``cell_*`` bounds.

    If ``show_progress`` is True and ``tqdm`` is installed, shows a bar over CAMS
    area cells (stderr).
    """
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.transform import xy as transform_xy
    from shapely.geometry import Polygon, box, mapping

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    m_area = base_mask & (st == 1)

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

    code_set = frozenset(int(c) for c in corine_codes)
    pop_exp = float(pop_exponent)
    floor = max(0.0, float(pop_floor))
    ls_resampling = (
        Resampling.bilinear
        if landscan_resampling == "bilinear"
        else Resampling.nearest
    )

    rows: list[dict] = []

    with rasterio.open(corine_path) as corine_src, rasterio.open(landscan_path) as ls_src:
        if corine_band < 1 or corine_band > int(corine_src.count):
            raise ValueError(f"corine_band {corine_band} invalid for {corine_path}")
        corine_crs = corine_src.crs
        if corine_crs is None:
            raise ValueError(f"CORINE raster has no CRS: {corine_path}")
        if ls_src.crs is None:
            raise ValueError(f"LandScan raster has no CRS: {landscan_path}")

        _area_idx = np.flatnonzero(m_area)
        _iter_cells = _area_idx
        if show_progress:
            try:
                from tqdm import tqdm

                _iter_cells = tqdm(
                    _area_idx,
                    desc="CAMS area cells (CORINE/LandScan)",
                    unit="cell",
                    total=int(_area_idx.size),
                    file=sys.stderr,
                )
            except ImportError:
                pass

        for i in _iter_cells:
            li, ji = int(lon_ii[i]), int(lat_ii[i])
            west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
            south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west

            if domain_bbox_wgs84 is not None and not _cams_cell_overlaps_bbox(
                west, south, east, north, domain_bbox_wgs84
            ):
                continue

            poly4326 = gpd.GeoDataFrame(
                geometry=[box(west, south, east, north)], crs="EPSG:4326"
            )
            geom3035 = poly4326.to_crs(corine_crs).geometry.iloc[0]
            if geom3035.is_empty:
                continue

            try:
                corine_win, corine_tr = rio_mask(
                    corine_src,
                    [mapping(geom3035)],
                    crop=True,
                    indexes=int(corine_band),
                )
            except ValueError:
                continue
            cw = corine_win.astype(np.float64)
            corine_arr = cw[0] if cw.ndim == 3 else cw
            nodata = corine_src.nodata
            if nodata is not None:
                corine_arr = np.where(corine_arr == float(nodata), np.nan, corine_arr)

            h, w = corine_arr.shape
            if h < 1 or w < 1:
                continue

            pop_dst = np.full((h, w), np.nan, dtype=np.float64)
            left, bottom, right, top = rasterio.transform.array_bounds(h, w, corine_tr)
            l0, b0, r0, t0 = transform_bounds(
                corine_crs, ls_src.crs, left, bottom, right, top, densify_pts=21
            )
            win_ls = from_bounds(l0, b0, r0, t0, transform=ls_src.transform).intersection(
                Window(0, 0, ls_src.width, ls_src.height)
            )
            if win_ls.width < 1 or win_ls.height < 1:
                pop_dst.fill(0.0)
            else:
                ls_arr = ls_src.read(1, window=win_ls).astype(np.float64)
                ls_tr = ls_src.window_transform(win_ls)
                ls_nodata = ls_src.nodata
                if ls_nodata is not None:
                    ls_arr = np.where(ls_arr == float(ls_nodata), np.nan, ls_arr)
                reproject(
                    source=ls_arr,
                    destination=pop_dst,
                    src_transform=ls_tr,
                    src_crs=ls_src.crs,
                    dst_transform=corine_tr,
                    dst_crs=corine_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=ls_resampling,
                )

            ok_data = np.isfinite(corine_arr)
            rint = np.zeros_like(corine_arr, dtype=np.int32)
            rint[ok_data] = np.rint(corine_arr[ok_data]).astype(np.int32)
            corine_hit = ok_data & np.isin(rint, list(code_set))

            pop_safe = np.maximum(np.nan_to_num(pop_dst, nan=0.0), floor)
            w_pix = np.where(corine_hit, np.power(pop_safe, pop_exp), 0.0)
            s = float(np.sum(w_pix))
            basis = "corine121_pop"

            if s <= 0:
                if fallback_if_no_corine == "skip":
                    continue
                w_pix = np.where(ok_data, np.power(pop_safe, pop_exp), 0.0)
                s = float(np.sum(w_pix))
                basis = "pop_full_cell"
                if s <= 0:
                    w_pix = np.where(ok_data, 1.0, 0.0)
                    s = float(np.sum(w_pix))
                    basis = "uniform_cell"
                    if s <= 0:
                        continue

            share = w_pix / s
            ys, xs = np.nonzero(share > 0)
            for row, col in zip(ys, xs):
                sh = float(share[row, col])
                wr = float(w_pix[row, col])
                if sh <= 0:
                    continue
                corners = []
                for r_off, c_off in ((0, 0), (0, 1), (1, 1), (1, 0)):
                    x, y = transform_xy(
                        corine_tr, row + r_off, col + c_off, offset="ul"
                    )
                    corners.append((float(x), float(y)))
                poly_native = Polygon(corners)
                if not poly_native.is_valid:
                    poly_native = poly_native.buffer(0)
                geom = (
                    gpd.GeoDataFrame(geometry=[poly_native], crs=corine_crs)
                    .to_crs(4326)
                    .geometry.iloc[0]
                )
                cv = int(rint[row, col]) if ok_data[row, col] else -9999
                pop_v = float(pop_dst[row, col]) if np.isfinite(pop_dst[row, col]) else 0.0
                rows.append(
                    {
                        "cams_source_index": int(i),
                        "weight_raw": wr,
                        "weight_share": sh,
                        "landscan_pop": pop_v,
                        "corine_value": cv,
                        "in_corine_target": bool(corine_hit[row, col]),
                        "weight_basis": basis,
                        "lon_cams_centre": float(lon[i]),
                        "lat_cams_centre": float(lat[i]),
                        "cell_west": west,
                        "cell_south": south,
                        "cell_east": east,
                        "cell_north": north,
                        "geometry": geom,
                    }
                )

    cols = [
        "cams_source_index",
        "weight_raw",
        "weight_share",
        "landscan_pop",
        "corine_value",
        "in_corine_target",
        "weight_basis",
        "lon_cams_centre",
        "lat_cams_centre",
        "cell_west",
        "cell_south",
        "cell_east",
        "cell_north",
        "geometry",
    ]
    if not rows:
        return gpd.GeoDataFrame(columns=cols, crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def downscale_cams_area_to_corine_landscan(
    ds: "xr.Dataset",
    base_mask: np.ndarray,
    pollutant: str,
    *,
    corine_path: Path,
    landscan_path: Path,
    corine_codes: tuple[int, ...] = (121, 3),
    corine_band: int = 1,
    pop_exponent: float = 1.0,
    pop_floor: float = 0.0,
    landscan_resampling: Literal["bilinear", "nearest"] = "bilinear",
    min_pixel_emission_kg: float = 0.0,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"] = "pop_in_cell",
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
    show_progress: bool = False,
) -> "gpd.GeoDataFrame":
    """
    Optional: same weights as ``build_cams_area_corine_landscan_weights``, then
    ``emission_kg_yr = weight_share * E`` per CAMS cell total *E* from ``pollutant``.
    """
    gdf = build_cams_area_corine_landscan_weights(
        ds,
        base_mask,
        corine_path=corine_path,
        landscan_path=landscan_path,
        corine_codes=corine_codes,
        corine_band=corine_band,
        pop_exponent=pop_exponent,
        pop_floor=pop_floor,
        landscan_resampling=landscan_resampling,
        fallback_if_no_corine=fallback_if_no_corine,
        domain_bbox_wgs84=domain_bbox_wgs84,
        show_progress=show_progress,
    )
    if gdf.empty:
        gdf = gdf.copy()
        gdf["emission_kg_yr"] = np.array([], dtype=np.float64)
        return gdf

    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    idx = gdf["cams_source_index"].to_numpy(dtype=np.int64)
    E = vals[idx]
    alloc = gdf["weight_share"].to_numpy(dtype=np.float64) * E
    if min_pixel_emission_kg > 0:
        alloc = np.where(alloc < min_pixel_emission_kg, 0.0, alloc)
        gdf = gdf.copy()
        gdf["emission_kg_yr"] = alloc
        for src in np.unique(idx):
            m = idx == src
            tot_E = float(vals[src])
            s2 = float(np.sum(gdf.loc[m, "emission_kg_yr"]))
            if s2 > 0 and s2 < tot_E and np.isfinite(tot_E) and tot_E > 0:
                gdf.loc[m, "emission_kg_yr"] *= tot_E / s2
    else:
        gdf = gdf.copy()
        gdf["emission_kg_yr"] = alloc
    return gdf


# EEA zip often unpacks a folder named with U+2018 (') not the ASCII letters "u2018".
_CORINE_DEMO_CANDIDATES = [
    Path("data/CORINE/U2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
    Path("data/CORINE/\u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"),
    Path("Input/CORINE/U2018_CLC2018_V2020_20u1.tif"),
]


def _first_existing_file(root: Path, candidates: list[Path]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p
    return None


def print_landscan_weight_diagnostics(
    gdf: "gpd.GeoDataFrame",
    *,
    landscan_path: Path | None = None,
    pop_floor: float = 0.0,
) -> None:
    """
    Explain why weight_share can look uniform: LandScan range, pop_floor capping,
    bilinear smoothing. Prints to stderr.
    """
    if gdf.empty:
        print("Diagnostics: empty GeoDataFrame.", file=sys.stderr)
        return
    P = gdf["landscan_pop"].to_numpy(dtype=np.float64)
    wr = gdf["weight_raw"].to_numpy(dtype=np.float64)
    sh = gdf["weight_share"].to_numpy(dtype=np.float64)

    print("\n=== LandScan / weight diagnostics ===", file=sys.stderr)
    if landscan_path is not None and Path(landscan_path).is_file():
        import rasterio

        with rasterio.open(landscan_path) as src:
            tr = src.transform
            print(
                f"LandScan file: {landscan_path.name}  CRS={src.crs}  "
                f"shape={src.width}x{src.height}  dtype={src.dtypes[0]}",
                file=sys.stderr,
            )
            print(
                f"  transform (approx. cell width in CRS units x): {abs(tr.a):.8g}",
                file=sys.stderr,
            )

    print(
        f"landscan_pop P over all output rows: min={float(np.nanmin(P)):.6g}  "
        f"max={float(np.nanmax(P)):.6g}  median={float(np.nanmedian(P)):.6g}",
        file=sys.stderr,
    )
    if pop_floor > 0:
        pct = 100.0 * float(np.mean(P < pop_floor))
        print(
            f"pop_floor={pop_floor}: {pct:.1f}% of rows have raw P < floor "
            f"(max(P,floor) makes those weights identical before **exponent).",
            file=sys.stderr,
        )
    z = float(np.mean(P <= 0))
    print(
        f"fraction of rows with P<=0 (after warp): {z:.3f}",
        file=sys.stderr,
    )

    cv_list: list[float] = []
    spread_list: list[float] = []
    for _, grp in gdf.groupby("cams_source_index", sort=False):
        p = grp["landscan_pop"].to_numpy(dtype=np.float64)
        s = grp["weight_share"].to_numpy(dtype=np.float64)
        m = float(np.mean(p))
        cv_list.append(float(np.std(p) / m) if m > 1e-12 else 0.0)
        pos = s[s > 0]
        if pos.size > 1:
            spread_list.append(float(np.max(pos) / np.min(pos)))
        elif pos.size == 1:
            spread_list.append(1.0)
    if cv_list:
        print(
            f"Per CAMS cell — median CV(std/mean) of P among output pixels: "
            f"{float(np.median(cv_list)):.4f}  (near 0 => bilinear smoothed field)",
            file=sys.stderr,
        )
    if spread_list:
        print(
            f"Per CAMS cell — median max(weight_share)/min(weight_share): "
            f"{float(np.median(spread_list)):.4f}  (near 1 => almost uniform shares)",
            file=sys.stderr,
        )

    print(
        f"Global weight_share: min={float(np.min(sh)):.6g} max={float(np.max(sh)):.6g}",
        file=sys.stderr,
    )
    print(
        f"Global weight_raw: min={float(np.min(wr)):.6g} max={float(np.max(wr)):.6g}",
        file=sys.stderr,
    )
    print(
        "Hint: use --pop-floor 0 (default now) if P is often < 1; "
        "try --landscan-resampling nearest for sharper P on 100 m cells.\n",
        file=sys.stderr,
    )


def export_cams_area_weights(
    gdf: "gpd.GeoDataFrame",
    *,
    geojson_path: Path,
    csv_path: Path | None = None,
) -> None:
    """Write weights to GeoJSON; optional CSV (lat/lon + attributes, no geometry column)."""
    geojson_path = Path(geojson_path)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    gdf.to_file(geojson_path, driver="GeoJSON")
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rp = gdf.geometry.representative_point()
        out = gdf.assign(lon=rp.x, lat=rp.y)
        cols = [c for c in out.columns if c != "geometry"]
        out[cols].to_csv(csv_path, index=False)


def write_weights_manifest(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_weights_method_report(path: Path, payload: dict[str, Any]) -> None:
    """Human-readable methodology (Markdown). User-requested report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = payload.get("corine_codes", [121, 3])
    codes_s = ", ".join(str(c) for c in codes)
    text = f"""# CAMS area weight proxy — method

Generated (UTC): `{payload.get("generated_utc", "")}`

## Purpose

Spatial **weights** for downscaling CAMS-REG **area** GNFR A (public power) emissions
onto fine-scale pixels. Emissions are **not** applied here; each CAMS grid cell gets
a set of non-negative weights that sum to 1 over CORINE raster pixels inside that cell.

## Input data

| Item | Role |
|------|------|
| CAMS NetCDF | `{payload.get("cam_netcdf", "")}` — `longitude_bounds` / `latitude_bounds` define each area cell; `longitude_source` / `latitude_source` are cell centres. |
| CORINE GeoTIFF | `{payload.get("corine_geotiff", "")}` — classification band **{payload.get("corine_band", 1)}** in the file CRS (e.g. EPSG:3035). |
| LandScan GeoTIFF | `{payload.get("landscan_geotiff", "")}` — population count; warped **bilinear** onto the CORINE grid for each CAMS cell window. |

## Domain filter

WGS84 bounding box (west, south, east, north): `{payload.get("domain_bbox_wgs84", [])}`  
Only CAMS area cells whose **rectangle intersects** this box are processed.

## Geometry of each weight row

Each feature is a **polygon**: the CORINE raster pixel footprint (four corners from
the CORINE affine transform, upper-left convention), reprojected to **EPSG:4326** for export.

## Weight construction (per CAMS area cell)

Conceptually:

- **CORINE** defines **where** downscaling is allowed (eligible pixels for public-power
  area mass). Outside the chosen classes, weight is **zero** — CORINE does not scale
  intensity, it **gates** pixels.
- **LandScan** defines **how much** of the cell total each **eligible** pixel receives:
  higher warped population value **P** ⇒ higher raw weight (for default
  `pop_exponent = 1`, weight is proportional to `max(P, pop_floor)`).

Steps in code:

1. Clip CORINE to the CAMS cell polygon in CORINE CRS (`rasterio.mask`).
2. Warp LandScan (bilinear) onto that **same** CORINE grid.
3. Let **P** = LandScan at each pixel, **C** = rounded CORINE class.
4. **Eligibility:** `E = 1` if `C` is in `{{{codes_s}}}`, else `E = 0`.
5. **Raw weight:** `w = E * max(P, pop_floor) ** pop_exponent`. So only eligible pixels
   can be positive; among them, larger **P** ⇒ larger **w** (monotone in **P** when
   `pop_exponent > 0`).
6. **Normalize:** `weight_share = w / sum(w)` over the cell (sum of shares = **1**).

**Note:** LandScan is a **count** per its source grid cell, not population **density**
(people per m²). Warping it onto 100 m CORINE pixels with **bilinear** interpolation
tends to **smooth** values: neighbouring pixels get similar **P**, so
`weight_share` can look almost uniform inside a CAMS cell. Use **`nearest`**
resampling (see manifest `landscan_resampling`) for a blockier, more contrasted field.

**`pop_floor` warning:** `w = max(P, pop_floor) ** exponent`. If `pop_floor` is large
(e.g. **1.0**) while many warped **P** are below 1 (common with bilinear fractions),
almost every pixel gets the same raw weight → **nearly uniform shares**. Prefer
`pop_floor=0` unless you deliberately want a minimum weight for low-pop pixels.

### Parameters used

- `corine_codes`: `{codes}`
- `corine_band`: {payload.get("corine_band", 1)}
- `pop_exponent`: {payload.get("pop_exponent", 1.0)}
- `pop_floor`: {payload.get("pop_floor", 0.0)}
- `landscan_resampling`: `{payload.get("landscan_resampling", "bilinear")}`
- `fallback_if_no_corine`: `{payload.get("fallback_if_no_corine", "")}`

### Fallbacks (when no pixel matches CORINE codes in the cell)

- **`pop_in_cell`**: Recompute `w` using **all** valid CORINE pixels in the cell (population-only weighting).
- If still no mass: **uniform** over valid CORINE pixels (`uniform_cell`).
- **`skip`**: Omit the cell entirely.

The column `weight_basis` records which case applied.

## Outputs

- GeoJSON: `{payload.get("weight_geojson", "")}` — polygons + attributes.
- Manifest JSON (machine-readable): same stem as GeoJSON with `_manifest.json`.
- This report: same stem with `_report.md`.

## Map

`python PublicPower/Auxiliaries/visualize_cams_area_weights_map.py` loads the manifest
to overlay **LandScan** and **CORINE target-class** rasters for comparison.

"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if xr is None:
        raise SystemExit("xarray is required.")
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(
        description="Build CAMS area CORINE+LandScan weights (Athens bbox demo) and export.",
    )
    ap.add_argument(
        "--out-geojson",
        type=Path,
        default=root / "PublicPower" / "outputs" / "cams_area_weights_athens.geojson",
        help="Output GeoJSON path",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV path (default: same stem as --out-geojson)",
    )
    ap.add_argument("--no-csv", action="store_true", help="Do not write CSV")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=None,
        help="WGS84 bbox; default: built-in Athens extent",
    )
    ap.add_argument("--no-export", action="store_true", help="Only print summary, no files")
    ap.add_argument("--pop-exponent", type=float, default=1.0)
    ap.add_argument(
        "--pop-floor",
        type=float,
        default=0.0,
        help="max(P, floor) before **exponent; values >0 flatten small P (avoid 1.0 if P often <1)",
    )
    ap.add_argument("--corine-band", type=int, default=1)
    ap.add_argument(
        "--landscan-resampling",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="Warp LandScan to CORINE grid: bilinear=smooth, nearest=blockier contrast",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print LandScan stats and per-cell weight spread to stderr",
    )
    args = ap.parse_args()

    nc = root / "data" / "given_CAMS" / "CAMS-REG-ANT_v8.1_TNO_ftp" / "netcdf" / "CAMS-REG-v8_1_emissions_year2019.nc"
    corine = _first_existing_file(root, _CORINE_DEMO_CANDIDATES)
    ls = root / "data" / "PublicPower" / "landscan" / "landscan-global-2020.tif"
    if not nc.is_file():
        print(f"Demo skipped (NetCDF not found): {nc}", file=sys.stderr)
        return
    if corine is None:
        print(
            "Demo skipped (CORINE GeoTIFF not found). Tried:",
            file=sys.stderr,
        )
        for rel in _CORINE_DEMO_CANDIDATES:
            print(f"  {root / rel}", file=sys.stderr)
        return
    if not ls.is_file():
        print(f"Demo skipped (LandScan not found): {ls}", file=sys.stderr)
        return

    bbox = tuple(args.bbox) if args.bbox is not None else ATHENS_BBOX_WGS84

    ds = xr.open_dataset(nc)
    try:
        mask = build_cams_area_mask_grc(ds)
        gdf = build_cams_area_corine_landscan_weights(
            ds,
            mask,
            corine_path=corine,
            landscan_path=ls,
            corine_band=int(args.corine_band),
            pop_exponent=float(args.pop_exponent),
            pop_floor=float(args.pop_floor),
            landscan_resampling=str(args.landscan_resampling),
            domain_bbox_wgs84=bbox,
        )
        pd = __import__("pandas")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(gdf.head(8).to_string())
        print()
        print("rows", len(gdf))
        if len(gdf):
            chk = gdf.groupby("cams_source_index", sort=False)["weight_share"].sum()
            print(
                "weight_share sum per cell (min/max):",
                float(chk.min()),
                float(chk.max()),
            )
            print("unique CAMS area cells:", gdf["cams_source_index"].nunique())
            print("weight_basis counts:\n", gdf["weight_basis"].value_counts().to_string())
        if args.diagnostics and len(gdf):
            print_landscan_weight_diagnostics(
                gdf,
                landscan_path=ls,
                pop_floor=float(args.pop_floor),
            )
        if not args.no_export and len(gdf):
            out_gj = args.out_geojson
            if not out_gj.is_absolute():
                out_gj = root / out_gj
            if args.no_csv:
                csv_p = None
            elif args.out_csv is not None:
                csv_p = args.out_csv
                if not csv_p.is_absolute():
                    csv_p = root / csv_p
            else:
                csv_p = out_gj.with_suffix(".csv")
            export_cams_area_weights(gdf, geojson_path=out_gj, csv_path=csv_p)
            print(f"Wrote {out_gj}")
            if csv_p is not None:
                print(f"Wrote {csv_p}")
            manifest_path = out_gj.with_name(out_gj.stem + "_manifest.json")
            report_path = out_gj.with_name(out_gj.stem + "_report.md")
            payload: dict[str, Any] = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "cam_netcdf": str(nc.resolve()),
                "corine_geotiff": str(corine.resolve()),
                "landscan_geotiff": str(ls.resolve()),
                "domain_bbox_wgs84": [float(x) for x in bbox],
                "corine_codes": [121, 3],
                "corine_band": int(args.corine_band),
                "pop_exponent": float(args.pop_exponent),
                "pop_floor": float(args.pop_floor),
                "landscan_resampling": str(args.landscan_resampling),
                "fallback_if_no_corine": "pop_in_cell",
                "weight_geojson": str(out_gj.resolve()),
                "map_overlay": {"grid_width": 950, "grid_height": 820},
            }
            write_weights_manifest(manifest_path, payload)
            write_weights_method_report(report_path, payload)
            print(f"Wrote {manifest_path}")
            print(f"Wrote {report_path}")
            print(
                "Map: python PublicPower/Auxiliaries/visualize_cams_area_weights_map.py"
            )
    finally:
        ds.close()


if __name__ == "__main__":
    main()
