#!/usr/bin/env python3
"""
Folium map: visualize a sector proxy GeoTIFF (weight raster) on a WGS84 grid.

Choose the raster with --tif, or --outputs-root + --preset for the standard build names.

Usage (from project root)::

  python SourceProxies/visualize_proxy_weights.py --preset agriculture
  python SourceProxies/visualize_proxy_weights.py --preset public_power_area --colour-mode per_cams_cell
  python SourceProxies/visualize_proxy_weights.py --preset residential --colour-mode per_cams_cell
  python SourceProxies/visualize_proxy_weights.py --preset solvents --clip-bbox 23.55,37.90,23.92,38.12 --colour-mode per_cams_cell --cams-sector gnfr_e
  python SourceProxies/visualize_proxy_weights.py --preset fugitive_area --clip-bbox 23.55,37.90,23.92,38.12 --colour-mode per_cams_cell --cams-sector gnfr_d
  python SourceProxies/visualize_proxy_weights.py --preset shipping --clip-bbox 23.55,37.90,23.92,38.12 --colour-mode per_cams_cell --cams-sector gnfr_g --cams-iso3 MED
  python SourceProxies/visualize_proxy_weights.py --preset offroad --clip-bbox 23.55,37.90,23.92,38.12 --colour-mode per_cams_cell --cams-sector gnfr_i
  python SourceProxies/visualize_proxy_weights.py --tif path/to/x.tif --colour-mode per_cams_cell --cams-nc ... --cams-sector public_power_a

Solvents preset maps ``Solvents/outputs/E_solvents_areasource.tif`` (GNFR E area weights). Use
``--clip-bbox west,south,east,north`` (WGS84) for a small view; if omitted, a default Attica-sized box is used.
Context layers (population, CORINE masks, OSM roads/landuse proxies) are recomputed on the cropped ref window.

Residential preset uses ``SourceProxies/outputs/EL/Residential_sourcearea.tif`` (multi-band proxy); if
missing, falls back to ``Residential/outputs/downscaled`` ``weights_<pol>.tif`` / ``emissions_<pol>.tif``.
Optional CORINE + Hotmaps + LandScan overlays from config.

The ``fugitive_area`` preset maps ``Fugitive_areasource.tif`` (GNFR D fugitive area weights, multi-band).
Context layers (population, ``P_pop``, per-group OSM coverage, CLC sector score, ``P_g``) are recomputed on the
same ref grid window as ``PROXY/config/fugitive/area_source.yaml`` (cropped to the map extent), matching the pipeline
inputs before CAMS-cell normalization.

Requires: folium, branca, rasterio, matplotlib, numpy; for ``per_cams_cell``: xarray, netCDF4
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path

import yaml

import numpy as np

try:
    from SourceProxies.corine_clc import corine_grid_to_weight_codes
    from SourceProxies.grid import first_existing_corine, iter_corine_search_paths
except ImportError:
    from corine_clc import corine_grid_to_weight_codes
    from grid import first_existing_corine, iter_corine_search_paths


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_reproject_helper():
    for sub in ("auxiliaries", "Auxiliaries"):
        p = _project_root() / "PublicPower" / sub / "greece_public_power_context_map.py"
        if p.is_file():
            break
    else:
        raise RuntimeError("greece_public_power_context_map.py not found under PublicPower/")
    spec = importlib.util.spec_from_file_location("_gpp_ctx_map", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._reproject_band_to_wgs84_grid


def _load_cams_a(root: Path):
    for sub in ("auxiliaries", "Auxiliaries"):
        p = root / "PublicPower" / sub / "cams_A_publicpower_greece.py"
        if p.is_file():
            break
    else:
        raise FileNotFoundError("cams_A_publicpower_greece.py not found")
    spec = importlib.util.spec_from_file_location("_cams_a", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cams_indices(ds):
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
    return lon_ii, lat_ii, lon_b, lat_b


def _mask_public_power_area(ds, iso3: str, root: Path) -> np.ndarray:
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == ca.IDX_A_PUBLIC_POWER) & ca._build_domain_mask(
        lon, lat, ci, ix, None
    )
    return base & (st == 1)


def _mask_agri_kl_area(ds, iso3: str, root: Path) -> np.ndarray:
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    kl = np.isin(emis, np.array([14, 15], dtype=np.int64))
    base = kl & ca._build_domain_mask(lon, lat, ci, ix, None)
    return base & (st == 1)


# GNFR C (other stationary combustion) = emission_category_index 3 (A=1, B=2, C=3, ...)
IDX_C_STATIONARY_COMBUSTION = 3


IDX_GNFR_E = 5
# TNO/CAMS-style ordering: A=1 … D=4 (fugitive / petroleum product emissions, etc.)
IDX_GNFR_D = 4
IDX_GNFR_G = 10
# GNFR I (off-road mobile machinery / pipelines / rail split) — CAMS-REG v8.1
IDX_GNFR_I = 12


def _mask_gnfr_e_area(ds, iso3: str, root: Path) -> np.ndarray:
    """CAMS GNFR E (solvents) area sources (source_type area = 1) for country."""
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_GNFR_E) & (st == 1) & (ci == ix)
    return base & ca._build_domain_mask(lon, lat, ci, ix, None)


def _mask_gnfr_d_area(ds, iso3: str, root: Path) -> np.ndarray:
    """CAMS GNFR D area sources (source_type area = 1) for country."""
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_GNFR_D) & (st == 1) & (ci == ix)
    return base & ca._build_domain_mask(lon, lat, ci, ix, None)


def _mask_gnfr_g_area(ds, iso3: str, root: Path) -> np.ndarray:
    """CAMS GNFR G (shipping) area sources (source_type area = 1) for country/region."""
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_GNFR_G) & (st == 1) & (ci == ix)
    return base & ca._build_domain_mask(lon, lat, ci, ix, None)


def _mask_gnfr_i_area(ds, iso3: str, root: Path) -> np.ndarray:
    """CAMS GNFR I (off-road) area sources (source_type area = 1) for country."""
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_GNFR_I) & (st == 1) & (ci == ix)
    return base & ca._build_domain_mask(lon, lat, ci, ix, None)


# GNFR J (waste) — emission_category_index 13 in CAMS-REG v8.1 (see Waste/j_waste_weights/config.yaml)
IDX_GNFR_J = 13


def _mask_gnfr_j_area(ds, iso3: str, root: Path) -> np.ndarray:
    """CAMS GNFR J (waste) area sources (source_type area = 1) for country."""
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_GNFR_J) & (st == 1) & (ci == ix)
    return base & ca._build_domain_mask(lon, lat, ci, ix, None)


def _mask_gnfr_c_area(ds, iso3: str, root: Path) -> np.ndarray:
    ca = _load_cams_a(root)
    ix = ca._country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    base = (emis == IDX_C_STATIONARY_COMBUSTION) & ca._build_domain_mask(
        lon, lat, ci, ix, None
    )
    return base & (st == 1)


def _cams_cell_id_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    ds,
    m_area: np.ndarray,
) -> np.ndarray:
    lon_ii, lat_ii, lon_b, lat_b = _cams_indices(ds)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    li = np.searchsorted(lon_b[:, 0], lons, side="right") - 1
    li = np.clip(li, 0, nlon - 1)
    ji = np.searchsorted(lat_b[:, 0], lats, side="right") - 1
    ji = np.clip(ji, 0, nlat - 1)
    valid_lon = (lons >= lon_b[li, 0]) & (lons <= lon_b[li, 1])
    valid_lat = (lats >= lat_b[ji, 0]) & (lats <= lat_b[ji, 1])
    in_bounds = valid_lon & valid_lat

    lookup = np.full(nlon * nlat, -1, dtype=np.int64)
    for i in sorted(int(x) for x in np.flatnonzero(m_area)):
        li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
        k = li_i * nlat + ji_i
        if lookup[k] < 0:
            lookup[k] = int(i)
    fk = li * nlat + ji
    cid = lookup[fk]
    return np.where(in_bounds & (cid >= 0), cid, -1)


def _normalize_weights_per_cams_cell(
    z: np.ndarray,
    cell_id: np.ndarray,
    *,
    base_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t01, display_valid) with t01 in [0,1] within each CAMS cell."""
    z = np.asarray(z)
    cell_id = np.asarray(cell_id)
    base_valid = np.asarray(base_valid, dtype=bool)
    if cell_id.shape != z.shape:
        if cell_id.size == z.size:
            cell_id = cell_id.reshape(z.shape)
        else:
            raise ValueError(
                f"cell_id shape {cell_id.shape} does not match z {z.shape} (sizes differ)"
            )
    out = np.full(z.shape, np.nan, dtype=np.float64)
    for cid in np.unique(cell_id):
        if cid < 0:
            continue
        m = (cell_id == cid) & base_valid
        if not np.any(m):
            continue
        vals = z[m]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            out[m] = 0.5
        else:
            out[m] = (z[m] - lo) / (hi - lo)
    disp = base_valid & np.isfinite(out)
    return out, disp


def _default_cams_nc(root: Path) -> Path | None:
    data = _load_sectors_config(root)
    if not data:
        return None
    p = data.get("paths", {}).get("cams_nc")
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def _load_sectors_config(root: Path) -> dict:
    cfg = root / "SourceProxies" / "config" / "sectors.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _sectors_ag_entry(data: dict) -> dict | None:
    for s in data.get("sectors", []):
        if str(s.get("builder", "")) == "agriculture_area":
            return s
    return None


def _resolve_corine_path(root: Path, data: dict) -> Path | None:
    configured = (data.get("paths") or {}).get("corine")
    try:
        return first_existing_corine(root, configured)
    except FileNotFoundError:
        return None


def _build_cams_grid_geojson(
    ds,
    m_area: np.ndarray,
    bbox_wgs84: tuple[float, float, float, float],
    lon_src: np.ndarray,
    lat_src: np.ndarray,
) -> dict:
    from shapely.geometry import box, mapping

    lon_ii, lat_ii, lon_b, lat_b = _cams_indices(ds)
    bw, bs, be, bn = bbox_wgs84
    features: list[dict] = []
    for i in np.flatnonzero(m_area):
        li, ji = int(lon_ii[i]), int(lat_ii[i])
        w, e = float(lon_b[li, 0]), float(lon_b[li, 1])
        s, n = float(lat_b[ji, 0]), float(lat_b[ji, 1])
        if s > n:
            s, n = n, s
        if w > e:
            w, e = e, w
        if e < bw or w > be or n < bs or s > bn:
            continue
        geom = box(w, s, e, n)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "cams_source_index": int(i),
                    "lon_c": float(lon_src[int(i)]),
                    "lat_c": float(lat_src[int(i)]),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _reproject_corine_to_wgs84_grid(
    corine_path: Path,
    corine_band: int,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    gh: int,
    gw: int,
) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds as window_from_bounds

    dst = np.full((gh, gw), -1, dtype=np.int32)
    with rasterio.open(corine_path) as src:
        if corine_band < 1 or corine_band > int(src.count):
            raise ValueError(f"CORINE band {corine_band} invalid")
        if src.crs is None:
            raise ValueError("CORINE has no CRS")
        lb, bb, rb, tb = transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north, densify_pts=21
        )
        win = window_from_bounds(lb, bb, rb, tb, transform=src.transform)
        win = win.round_lengths().intersection(Window(0, 0, src.width, src.height))
        if win.width < 1 or win.height < 1:
            return dst
        arr = src.read(corine_band, window=win).astype(np.float64)
        wt = src.window_transform(win)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        tmp = np.full((gh, gw), np.nan, dtype=np.float64)
        reproject(
            source=arr,
            destination=tmp,
            src_transform=wt,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        ok = np.isfinite(tmp)
        dst[ok] = np.rint(tmp[ok]).astype(np.int32)
    return dst


def _clc_ag_colors_vivid() -> dict[int, str]:
    """Evenly separated hues (golden angle) with high S/V for readable class patches."""
    import colorsys

    out: dict[int, str] = {}
    phi = 0.618033988749895
    for i, c in enumerate(range(12, 23)):
        h = (i * phi + 0.07) % 1.0
        s = 0.88 + (i % 3) * 0.035
        v = 0.94 + (i % 2) * 0.05
        r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), min(v, 1.0))
        out[c] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return out


CLC_AG_COLORS: dict[int, str] = _clc_ag_colors_vivid()

CLC_AG_NAMES: dict[int, str] = {
    12: "Non-irrigated arable",
    13: "Permanent crops",
    14: "Complex cultivation",
    15: "Land principally ag + nat veg",
    16: "Ag with nat veg (significant)",
    17: "Pasture / grassland",
    18: "Annual crops + perm crops",
    19: "Agro-forestry",
    20: "Complex mosaic (small fields)",
    21: "Pastures",
    22: "Other agriculture",
}

# CORINE raster L1 legend (1–44 scheme): only these three are drawn on residential maps.
RESIDENTIAL_CORINE_L1: dict[int, tuple[str, str]] = {
    1: ("Continuous urban fabric", "#c62828"),
    2: ("Discontinuous urban fabric", "#f9a825"),
    3: ("Industrial or commercial units", "#1565c0"),
}


def _hex_to_rgba_u8(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def _corine_ag_overlay_rgba(
    clc: np.ndarray,
    ag_codes: tuple[int, ...],
    *,
    opacity: int = 255,
    draw_class_edges: bool = True,
) -> np.ndarray:
    h, w = clc.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    ag_list = [int(x) for x in ag_codes]
    ag_set = frozenset(ag_list)
    ag_mask = np.isin(clc, list(ag_set))
    # Non-ag pixels stay transparent so the basemap shows (opaque gray filled the whole frame).
    for code in ag_codes:
        m = clc == int(code)
        if not np.any(m):
            continue
        r, g, b, a = _hex_to_rgba_u8(CLC_AG_COLORS.get(int(code), "#888888"), opacity)
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = a
    if draw_class_edges and len(ag_list) > 0:
        valid = np.isin(clc, list(ag_set))
        edge = np.zeros((h, w), dtype=bool)
        padded = np.full((h + 2, w + 2), -1, dtype=clc.dtype)
        padded[1 : h + 1, 1 : w + 1] = clc
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            nvalid = np.isin(nb, list(ag_set))
            edge |= valid & nvalid & (clc != nb)
        # Single-pixel boundaries only: thick/dilated edges merged into bogus "black regions".
        rgba[edge, 0] = 70
        rgba[edge, 1] = 70
        rgba[edge, 2] = 85
        rgba[edge, 3] = 160
    return rgba


def _rasterize_nuts2_wgs84(
    nuts_gpkg: Path,
    nuts_cntr: str,
    dst_transform,
    gh: int,
    gw: int,
) -> tuple[np.ndarray, list[str]]:
    import geopandas as gpd
    from rasterio import features

    nuts = gpd.read_file(nuts_gpkg)
    n2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    cc = n2["CNTR_CODE"].astype(str).str.strip().str.upper()
    n2 = n2[cc == str(nuts_cntr).strip().upper()].copy()
    if n2.empty:
        raise ValueError(f"No NUTS2 for CNTR_CODE={nuts_cntr!r}")
    n2 = n2.to_crs(4326)
    nuts_ids: list[str] = []
    shapes: list[tuple[object, int]] = []
    for k, (_, row) in enumerate(n2.iterrows()):
        nuts_ids.append(str(row["NUTS_ID"]).strip())
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        from shapely.geometry import mapping

        shapes.append((mapping(geom), k + 1))
    out = features.rasterize(
        shapes,
        out_shape=(gh, gw),
        transform=dst_transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )
    return out, nuts_ids


def _agriculture_wp_grid(
    clc: np.ndarray,
    nuts_r: np.ndarray,
    nuts_ids: list[str],
    weights_csv: Path,
    pollutant: str,
    ag_codes: tuple[int, ...],
) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(weights_csv)
    pol_u = str(pollutant).strip().upper()
    sub = df[df["pollutant"].astype(str).str.upper() == pol_u]
    if sub.empty:
        raise ValueError(f"No rows in weights_long for pollutant={pollutant!r}")

    nuts_to_ix = {nid: k + 1 for k, nid in enumerate(nuts_ids)}
    n_nuts = len(nuts_ids) + 1
    lookup = np.zeros((n_nuts, 23), dtype=np.float64)
    for _, r in sub.iterrows():
        nid = str(r["NUTS_ID"]).strip()
        ix = nuts_to_ix.get(nid)
        if ix is None:
            continue
        cl = int(r["CLC_CODE"])
        if 0 <= cl <= 22:
            lookup[ix, cl] = float(r["w_p"])

    h, w = clc.shape
    wp = np.zeros((h, w), dtype=np.float64)
    valid_nuts = (nuts_r >= 1) & (nuts_r < n_nuts)
    for cl in ag_codes:
        m = (clc == int(cl)) & valid_nuts
        if not np.any(m):
            continue
        ii = nuts_r[m]
        wp[m] = lookup[ii, int(cl)]
    return wp


def _agriculture_nuts_sum_sp_grid(
    nuts_r: np.ndarray,
    nuts_ids: list[str],
    weights_csv: Path,
    pollutant: str,
    ag_codes: tuple[int, ...],
) -> np.ndarray:
    """One value per NUTS2: sum of S_p over ag CLC rows (weights_long); fill whole region on display grid."""
    import pandas as pd

    df = pd.read_csv(weights_csv)
    pol_u = str(pollutant).strip().upper()
    sub = df[df["pollutant"].astype(str).str.upper() == pol_u]
    if sub.empty:
        raise ValueError(f"No rows in weights_long for pollutant={pollutant!r}")
    if "S_p" not in sub.columns:
        raise ValueError("weights_long.csv must contain column S_p for NUTS aggregate viz")
    ag_set = frozenset(int(x) for x in ag_codes)
    sub = sub[sub["CLC_CODE"].astype(int).isin(ag_set)]
    summed = sub.groupby(sub["NUTS_ID"].astype(str).str.strip())["S_p"].sum()
    nuts_val = {str(k).strip(): float(v) for k, v in summed.items() if float(v) > 0}

    h, w = nuts_r.shape
    out = np.zeros((h, w), dtype=np.float64)
    for k, nid in enumerate(nuts_ids):
        ix = k + 1
        v = nuts_val.get(str(nid).strip())
        if v is not None:
            out[nuts_r == ix] = v
    return out


def _warn_if_wp_invisible(
    wp: np.ndarray,
    *,
    pollutant: str,
    nuts_ids: list[str],
    weights_csv: Path,
    nuts_r: np.ndarray,
    clc_w: np.ndarray,
) -> None:
    import pandas as pd

    if np.any(np.isfinite(wp) & (wp > 0)):
        return
    df = pd.read_csv(weights_csv)
    pol_u = str(pollutant).strip().upper()
    sub = df[df["pollutant"].astype(str).str.upper() == pol_u]
    csv_nuts = {str(x).strip() for x in sub["NUTS_ID"].unique()}
    gpkg_nuts = set(nuts_ids)
    matched = csv_nuts & gpkg_nuts
    ag_px = int(np.sum((clc_w >= 12) & (clc_w <= 22)))
    nuts_px = int(np.sum(nuts_r > 0))
    overlap = int(np.sum((clc_w >= 12) & (clc_w <= 22) & (nuts_r > 0)))
    print(
        f"Warning: w_p overlay has no positive pixels (pollutant={pol_u!r}). "
        f"NUTS_ID overlap CSV↔GeoPackage: {len(matched)} regions; "
        f"pixels: ag_CLC={ag_px}, inside_NUTS2={nuts_px}, ag∩NUTS={overlap}. "
        f"If overlap is 0, check country.nuts_cntr vs GeoPackage CNTR_CODE.",
        file=sys.stderr,
    )


WP_IMPORTANCE_CMAP = "YlOrRd"


def _wp_global_log_bounds(
    wp_grids: list[np.ndarray],
    eps: float = 1e-18,
    clip_percentile: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    parts: list[np.ndarray] = []
    for wp in wp_grids:
        v = wp[np.isfinite(wp) & (wp > 0)]
        if v.size:
            parts.append(np.log10(np.maximum(v.astype(np.float64), eps)))
    if not parts:
        return None
    all_lv = np.concatenate(parts)
    if clip_percentile is not None:
        p_lo, p_hi = clip_percentile
        lo = float(np.percentile(all_lv, p_lo))
        hi = float(np.percentile(all_lv, p_hi))
    else:
        lo, hi = float(np.min(all_lv)), float(np.max(all_lv))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _wp_importance_rgba(
    wp: np.ndarray,
    *,
    cmap_name: str = WP_IMPORTANCE_CMAP,
    eps: float = 1e-18,
    log_vmin: float | None = None,
    log_vmax: float | None = None,
    alpha: int = 238,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    h, w = wp.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(wp) & (wp > 0)
    if not np.any(valid):
        return rgba
    lv = np.log10(np.maximum(wp[valid], eps))
    if log_vmin is None:
        lo = float(np.min(lv))
    else:
        lo = float(log_vmin)
    if log_vmax is None:
        hi = float(np.max(lv))
    else:
        hi = float(log_vmax)
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((lv - lo) / (hi - lo), 0.0, 1.0)
    cmap = colormaps[cmap_name]
    c = cmap(t)
    rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 3] = int(np.clip(alpha, 0, 255))
    return rgba


def _corine_residential_l1_overlay_rgba(
    clc: np.ndarray,
    *,
    opacity: int = 255,
    draw_class_edges: bool = True,
) -> np.ndarray:
    """Paint only codes 1, 2, 3 with fixed legend colours."""
    h, w = clc.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    focus = tuple(sorted(RESIDENTIAL_CORINE_L1.keys()))
    ag_set = frozenset(focus)
    for code in focus:
        m = clc == int(code)
        if not np.any(m):
            continue
        name_col = RESIDENTIAL_CORINE_L1.get(int(code), ("", "#888888"))
        hex_col = name_col[1]
        r, g, b, a = _hex_to_rgba_u8(hex_col, opacity)
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = a
    if draw_class_edges:
        valid = np.isin(clc, list(ag_set))
        edge = np.zeros((h, w), dtype=bool)
        padded = np.full((h + 2, w + 2), -1, dtype=clc.dtype)
        padded[1 : h + 1, 1 : w + 1] = clc
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            nvalid = np.isin(nb, list(ag_set))
            edge |= valid & nvalid & (clc != nb)
        rgba[edge, 0] = 40
        rgba[edge, 1] = 40
        rgba[edge, 2] = 55
        rgba[edge, 3] = 200
    return rgba


def _clc_urban_colors(codes: tuple[int, ...]) -> dict[int, str]:
    import colorsys

    phi = 0.618033988749895
    out: dict[int, str] = {}
    for i, c in enumerate(sorted(set(codes))):
        h = (i * phi + 0.55) % 1.0
        s = 0.55 + (i % 4) * 0.1
        v = 0.92
        r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), min(v, 1.0))
        out[int(c)] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return out


def _corine_urban_overlay_rgba(
    clc: np.ndarray,
    focus_codes: tuple[int, ...],
    *,
    opacity: int = 255,
    draw_class_edges: bool = True,
) -> np.ndarray:
    """Highlight selected CLC urban / peri-urban classes; other pixels transparent."""
    h, w = clc.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cols = _clc_urban_colors(focus_codes)
    ag_list = [int(x) for x in focus_codes]
    ag_set = frozenset(ag_list)
    for code in focus_codes:
        m = clc == int(code)
        if not np.any(m):
            continue
        r, g, b, a = _hex_to_rgba_u8(cols.get(int(code), "#888888"), opacity)
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = a
    if draw_class_edges and len(ag_list) > 0:
        valid = np.isin(clc, list(ag_set))
        edge = np.zeros((h, w), dtype=bool)
        padded = np.full((h + 2, w + 2), -1, dtype=clc.dtype)
        padded[1 : h + 1, 1 : w + 1] = clc
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            nvalid = np.isin(nb, list(ag_set))
            edge |= valid & nvalid & (clc != nb)
        rgba[edge, 0] = 55
        rgba[edge, 1] = 55
        rgba[edge, 2] = 70
        rgba[edge, 3] = 180
    return rgba


def _corine_class_grid_on_display_grid(
    corine_path: Path,
    corine_band: int,
    dst_t,
    gh: int,
    gw: int,
    reproject_one,
) -> np.ndarray:
    """
    Same WGS84 pixel grid and warp path as the main proxy raster (nearest neighbour for classes).
    """
    arr = reproject_one(
        corine_path,
        dst_t,
        (gh, gw),
        resampling="nearest",
        band=int(corine_band),
    )
    arr = np.asarray(arr, dtype=np.float64)
    clc = np.full((gh, gw), -1, dtype=np.int32)
    ok = np.isfinite(arr)
    if np.any(ok):
        clc[ok] = np.rint(arr[ok]).astype(np.int32)
    return clc


def _load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_under_root(root: Path, p: str | Path | None) -> Path | None:
    if p is None:
        return None
    x = Path(p)
    return x if x.is_absolute() else (root / x)


def _reproject_array_to_wgs84_grid(
    arr: np.ndarray,
    src_transform,
    src_crs: str,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str = "bilinear",
) -> np.ndarray:
    """Warp a single-band float array (ref CRS) onto the Folium WGS84 pixel grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    height, width = int(dst_shape[0]), int(dst_shape[1])
    dst = np.full((height, width), np.nan, dtype=np.float64)
    res = getattr(Resampling, str(resampling))
    reproject(
        source=np.asarray(arr, dtype=np.float64),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=res,
    )
    return dst


DEFAULT_SOLVENTS_CLIP_WGS84 = (23.55, 37.88, 23.95, 38.12)


def _intersect_ref_window_with_wgs84_bbox(
    ref: dict,
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict | None:
    """
    Build a ref-like profile for the intersection of the ref grid with a WGS84 bbox.
    Returns None if empty.
    """
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.windows import Window, from_bounds as win_from_bounds
    from rasterio.warp import transform_bounds

    crs = rasterio.crs.CRS.from_string(str(ref["crs"]))
    rw, rs, re, rn = (float(x) for x in ref["window_bounds_3035"])
    W, S, E, N = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    li = max(rw, min(W, E))
    ri = min(re, max(W, E))
    bi = max(rs, min(S, N))
    ti = min(rn, max(S, N))
    if li >= ri or bi >= ti:
        return None
    win = win_from_bounds(li, bi, ri, ti, transform=ref["transform"])
    win = win.round_lengths().intersection(Window(0, 0, int(ref["width"]), int(ref["height"])))
    if win.width < 2 or win.height < 2:
        return None
    mini_transform = rasterio.windows.transform(win, ref["transform"])
    ml, mb, mr, mt = array_bounds(int(win.height), int(win.width), mini_transform)
    return {
        **ref,
        "height": int(win.height),
        "width": int(win.width),
        "transform": mini_transform,
        "window_bounds_3035": (ml, mb, mr, mt),
        "domain_bbox_wgs84": (west, south, east, north),
    }


def _build_solvents_context_layers(
    root: Path,
    cfg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Recompute Solvents proxy ingredients on a cropped ref window, warp to WGS84 display grid.

    Returns (list of (layer_title, rgba uint8 HxWx4), extra HTML legend fragment).
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import json

        from Solvents.indicators.osm_pbf import load_osm_indicators
        from Solvents.indicators.population import warp_population_to_ref
        from Solvents.indicators.solvent_context import (
            build_solvent_context_masks,
            title_for_solvent_mask_key,
        )
        from Solvents.io.ref_grid import load_ref_profile
    except ImportError as exc:
        print(f"Warning: Solvents context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    cfgp = cfg_path if cfg_path.is_absolute() else (root / cfg_path)
    if not cfgp.is_file():
        print(f"Warning: solvents config not found: {cfgp}", file=sys.stderr)
        return [], ""

    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    ref = load_ref_profile(root, cfg)
    mini = _intersect_ref_window_with_wgs84_bbox(ref, west, south, east, north)
    if mini is None:
        print(
            "Warning: --clip-bbox does not overlap the Solvents reference grid.",
            file=sys.stderr,
        )
        return [], ""

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    layers_3035: list[tuple[str, np.ndarray]] = []

    pop = warp_population_to_ref(root, cfg["paths"].get("population_tif", ""), mini)
    layers_3035.append(("Population (ref window)", pop))

    try:
        masks = build_solvent_context_masks(root, cfg, mini)
    except Exception as exc:
        print(f"Warning: Solvents CORINE/OSM masks failed: {exc}", file=sys.stderr)
        masks = {}

    has_osm_gpkg = any(k.startswith("osm_") for k in masks)
    if masks:
        for key in sorted(masks.keys()):
            if key.startswith("corine_") or key.startswith("osm_"):
                arr = np.asarray(masks[key], dtype=np.float32)
                if np.any(arr > 0):
                    layers_3035.append((title_for_solvent_mask_key(key), arr))

    try:
        osm = load_osm_indicators(root, cfg, mini)
        road_layers = (
            ("road_length", "OSM · road_length (sampled)"),
            ("weighted_road_length", "OSM · weighted_road_length"),
        )
        for key, label in road_layers:
            arr = osm.get(key)
            if arr is not None and np.any(np.asarray(arr) > 0):
                layers_3035.append((label, np.asarray(arr, dtype=np.float32)))
        if not has_osm_gpkg:
            for key, label in (
                ("service_osm", "OSM landuse · service_osm (aggregate)"),
                ("industry_osm", "OSM landuse · industry_osm (aggregate)"),
            ):
                arr = osm.get(key)
                if arr is not None and np.any(np.asarray(arr) > 0):
                    layers_3035.append((label, np.asarray(arr, dtype=np.float32)))
    except Exception as exc:
        print(f"Warning: Solvents OSM road/aggregate layers skipped: {exc}", file=sys.stderr)

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            warped = _reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                dst_shape,
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Solvents proxy inputs (cropped window)</summary>
  <p class="pl-hint">Layers are recomputed on the ref grid clipped to the map extent (EPSG:3035 window).
  <b>CORINE</b>: one binary mask per <code>corine_codes</code> group. <b>OSM</b>: one mask per
  <code>solvent_family</code> in <code>paths.osm_solvent_gpkg</code> (layers <code>osm_solvent_polygons</code> /
  <code>osm_solvent_points</code>), same as <code>Solvents/indicators/osm_pbf.py</code>. If the GPKG is missing,
  aggregate PBF road/landuse proxies are shown instead. Population first; Viridis + 2–98% stretch; zeros transparent.</p>
</details>"""
    return out_layers, leg


def _build_fugitive_context_layers(
    root: Path,
    fugitive_cfg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
    sectors_data: dict | None,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Rebuild Fugitive proxy ingredients on a cropped ref window, warp to WGS84 display grid.

    Returns (list of (layer_title, rgba uint8 HxWx4), HTML legend fragment).
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import geopandas as gpd
        from rasterio.enums import Resampling

        from PROXY.core.dataloaders import load_yaml
        from PROXY.core.dataloaders.raster import warp_raster_to_ref
        from PROXY.core.osm_corine_proxy import build_all_group_pg, build_p_pop
        from PROXY.core.ref_profile import load_area_ref_profile, resolve_corine_path
    except ImportError as exc:
        print(f"Warning: Fugitive context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    cfgp = fugitive_cfg_path if fugitive_cfg_path.is_absolute() else (root / fugitive_cfg_path)
    if not cfgp.is_file():
        print(f"Warning: fugitive config not found: {cfgp}", file=sys.stderr)
        return [], ""

    try:
        fugitive_cfg: dict = load_yaml(cfgp)
    except Exception as exc:
        print(f"Warning: fugitive config read failed: {exc}", file=sys.stderr)
        return [], ""
    fugitive_cfg["_project_root"] = root
    fugitive_cfg["_config_path"] = cfgp

    paths = dict(fugitive_cfg.get("paths") or {})
    if sectors_data:
        main_paths = sectors_data.get("paths") or {}
        for key in ("cams_nc", "nuts_gpkg", "corine"):
            if key in main_paths:
                paths[key] = main_paths[key]
    fugitive_cfg["paths"] = paths

    try:
        ref = load_area_ref_profile(fugitive_cfg)
    except Exception as exc:
        print(f"Warning: Fugitive ref grid failed: {exc}", file=sys.stderr)
        return [], ""

    mini = _intersect_ref_window_with_wgs84_bbox(ref, west, south, east, north)
    if mini is None:
        print(
            "Warning: map extent does not overlap the Fugitive reference grid.",
            file=sys.stderr,
        )
        return [], ""

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])

    def _rp(p: str | None) -> Path | None:
        if not p:
            return None
        x = Path(p)
        return x if x.is_absolute() else (root / x)

    cor_p = _rp(paths.get("corine"))
    if cor_p is None or not cor_p.is_file():
        try:
            if paths.get("corine"):
                cor_p = resolve_corine_path(root, paths["corine"])
            else:
                alt = (sectors_data or {}).get("paths", {}).get("corine")
                if alt:
                    cor_p = resolve_corine_path(root, alt)
        except (FileNotFoundError, TypeError, OSError, ValueError):
            cor_p = None
    pop_p = _rp(paths.get("population_tif"))
    osm_p = _rp(paths.get("osm_fugitive_gpkg"))
    gy = _rp(paths.get("ceip_groups_yaml"))

    if cor_p is None or not cor_p.is_file():
        print("Warning: Fugitive CORINE path missing; context layers skipped.", file=sys.stderr)
        return [], ""
    if pop_p is None or not pop_p.is_file():
        print("Warning: Fugitive population_tif missing; context layers skipped.", file=sys.stderr)
        return [], ""
    if osm_p is None or not osm_p.is_file():
        print("Warning: Fugitive OSM GPKG missing; context layers skipped.", file=sys.stderr)
        return [], ""
    if gy is None or not gy.is_file():
        print("Warning: Fugitive ceip_groups_yaml missing; context layers skipped.", file=sys.stderr)
        return [], ""

    try:
        clc = warp_raster_to_ref(
            cor_p,
            mini,
            band=1,
            resampling=Resampling.nearest,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        clc_nn = np.full(clc.shape, -9999, dtype=np.int32)
        _m = np.isfinite(clc)
        clc_nn[_m] = np.rint(clc[_m]).astype(np.int32, copy=False)
        pop = warp_raster_to_ref(
            pop_p,
            mini,
            band=1,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=np.nan,
        )
        p_pop = build_p_pop(pop, mini)
        with gy.open(encoding="utf-8") as gf:
            group_specs_root = yaml.safe_load(gf) or {}
        groups_raw: dict = dict(group_specs_root.get("groups") or {})
        osm_gdf = gpd.read_file(osm_p)
        pcfg = fugitive_cfg.get("proxy") or {}
        group_pg = build_all_group_pg(
            clc_nn,
            osm_gdf,
            {"groups": groups_raw},
            mini,
            pcfg,
            p_pop,
        )
    except Exception as exc:
        print(f"Warning: Fugitive proxy rebuild failed: {exc}", file=sys.stderr)
        return [], ""

    layers_3035: list[tuple[str, np.ndarray]] = [
        ("Fugitive · population (warped, ref window)", np.asarray(pop, dtype=np.float32)),
        ("Fugitive · P_pop (z-score of pop. density)", np.asarray(p_pop, dtype=np.float32)),
    ]
    for gid in ("G1", "G2", "G3", "G4"):
        d = group_pg.get(gid) or {}
        o = d.get("osm_raw")
        c = d.get("clc_raw")
        pg = d.get("p_g")
        if o is not None:
            layers_3035.append((f"Fugitive · {gid} OSM coverage (0–1)", np.asarray(o, dtype=np.float32)))
        if c is not None:
            layers_3035.append((f"Fugitive · {gid} CLC sector score", np.asarray(c, dtype=np.float32)))
        if pg is not None:
            layers_3035.append(
                (f"Fugitive · {gid} P_g proxy (pre–CAMS-cell norm)", np.asarray(pg, dtype=np.float32))
            )

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            warped = _reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip Fugitive layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Fugitive proxy inputs (cropped ref window)</summary>
  <p class="pl-hint">Layers match <code>PROXY/sectors/D_Fugitive/pipeline.py</code> on the fine grid clipped to the map extent
  (YAML from <code>--fugitive-config</code>). Viridis + 2–98% stretch; zeros transparent. <code>P_g</code> is the
  per-group blend before within–CAMS-cell normalization of final weights.</p>
</details>"""
    return out_layers, leg


def _build_shipping_context_layers(
    root: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Rebuild Shipping proxy ingredients on a 3035 grid over the map bbox, warp to WGS84 display grid.

    Does not require the built ``Shipping_areasource.tif``; uses default data paths under ``data/Shipping/``
    and ``first_existing_corine`` when available.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from rasterio.transform import from_bounds as rio_from_bounds
        from rasterio.warp import transform_bounds

        from Shipping.shipping_areasource import build_combined_proxy, resolve_corine_tif
    except ImportError as exc:
        print(f"Warning: Shipping context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    emodnet_p = root / "data" / "Shipping" / "EMODnet" / "EMODnet_HA_Vessel_Density_allAvg" / "vesseldensity_all_2019.tif"
    osm_p = root / "data" / "Shipping" / "osm_shipping_layers.gpkg"
    try:
        cor_p = first_existing_corine(root, None)
    except (FileNotFoundError, TypeError, OSError) as exc:
        print(f"Warning: Shipping CORINE not found ({exc}); context layers skipped.", file=sys.stderr)
        return [], ""

    if not emodnet_p.is_file() or not osm_p.is_file():
        print(
            "Warning: Shipping EMODnet or OSM default path missing; context layers skipped.",
            file=sys.stderr,
        )
        return [], ""

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", "EPSG:3035", west, south, east, north, densify_pts=21
    )
    res_m = 100.0
    w = max(1, int(np.ceil((right - left) / res_m)))
    h = max(1, int(np.ceil((top - bottom) / res_m)))
    tr = rio_from_bounds(left, bottom, right, top, width=w, height=h)
    ref: dict = {
        "height": h,
        "width": w,
        "transform": tr,
        "crs": "EPSG:3035",
        "window_bounds_3035": (float(left), float(bottom), float(right), float(top)),
    }
    try:
        _, diag = build_combined_proxy(
            ref,
            emodnet_path=emodnet_p,
            corine_path=resolve_corine_tif(Path(cor_p)),
            osm_gpkg=osm_p,
        )
    except Exception as exc:
        print(f"Warning: Shipping proxy rebuild failed: {exc}", file=sys.stderr)
        return [], ""

    crs_str = "EPSG:3035"
    transform = ref["transform"]
    layers_3035: list[tuple[str, np.ndarray]] = [
        ("Shipping · EMODnet raw (warped)", np.asarray(diag["emodnet_raw"], dtype=np.float32)),
        ("Shipping · D_n damped", np.asarray(diag["D_n_damped"], dtype=np.float32)),
        ("Shipping · OSM coverage (0–1)", np.asarray(diag["osm_coverage"], dtype=np.float32)),
        ("Shipping · CLC port fraction", np.asarray(diag["clc_port_frac"], dtype=np.float32)),
        ("Shipping · z(OSM)", np.asarray(diag["z_osm"], dtype=np.float32)),
        ("Shipping · z(CLC port)", np.asarray(diag["z_clc"], dtype=np.float32)),
    ]

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            warped = _reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip Shipping layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Shipping proxy inputs (bbox 3035 @100 m)</summary>
  <p class="pl-hint">Rebuilt from EMODnet, OSM port layers, and CORINE port class (123 / CLC44 5) on a coarse
  grid over the map extent — for comparison with <code>Shipping_areasource.tif</code> on your real fine grid.
  Viridis + 2–98% stretch; zeros transparent.</p>
</details>"""
    return out_layers, leg


def _shipping_legend_block() -> str:
    return """
<details class="pl-details">
  <summary>Shipping (GNFR G)</summary>
  <p class="pl-hint">Context layers: EMODnet vessel density, OSM port coverage, CORINE port fraction, and z-scored OSM/CLC.</p>
</details>"""


def _build_offroad_context_layers(
    root: Path,
    offroad_cfg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
    sectors_data: dict | None,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Rebuild Offroad proxy ingredients on a 3035 grid over the map bbox, warp to WGS84 display grid.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from rasterio.transform import from_bounds as rio_from_bounds
        from rasterio.warp import transform_bounds

        from Offroad.offroad_areasource import diagnostics_for_extent
    except ImportError as exc:
        print(f"Warning: Offroad context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    if not offroad_cfg_path.is_file():
        return [], ""

    with offroad_cfg_path.open(encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    paths = dict(yaml_cfg.get("paths") or {})
    if sectors_data:
        main_paths = sectors_data.get("paths") or {}
        for key in ("cams_nc", "corine", "nuts_gpkg"):
            if key in main_paths:
                paths[key] = main_paths[key]
        if main_paths.get("landscan"):
            paths["population_tif"] = main_paths["landscan"]

    gh, gw = int(dst_shape[0]), int(dst_shape[1])
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", "EPSG:3035", west, south, east, north, densify_pts=21
    )
    res_m = 100.0
    w = max(1, int(np.ceil((right - left) / res_m)))
    h = max(1, int(np.ceil((top - bottom) / res_m)))
    tr = rio_from_bounds(left, bottom, right, top, width=w, height=h)
    ref: dict = {
        "height": h,
        "width": w,
        "transform": tr,
        "crs": "EPSG:3035",
        "window_bounds_3035": (float(left), float(bottom), float(right), float(top)),
    }
    try:
        diag = diagnostics_for_extent(root, yaml_cfg, paths, ref)
    except Exception as exc:
        print(f"Warning: Offroad proxy diagnostics failed: {exc}", file=sys.stderr)
        return [], ""

    crs_str = "EPSG:3035"
    transform = ref["transform"]
    layers_3035: list[tuple[str, np.ndarray]] = [
        ("Offroad · z(rail)", np.asarray(diag["z_rail"], dtype=np.float32)),
        ("Offroad · z(pipeline)", np.asarray(diag["z_pipeline"], dtype=np.float32)),
        ("Offroad · z(agri CLC)", np.asarray(diag["z_agri"], dtype=np.float32)),
        ("Offroad · z(industry combined)", np.asarray(diag["z_industry_combined"], dtype=np.float32)),
        ("Offroad · z(pop non-road)", np.asarray(diag["z_pop_nonroad"], dtype=np.float32)),
        ("Offroad · P_nonroad raw (pre-shares)", np.asarray(diag["p_nonroad_raw"], dtype=np.float32)),
        ("Offroad · rail coverage (0–1)", np.asarray(diag["rail_coverage_raw"], dtype=np.float32)),
        ("Offroad · pipeline coverage (0–1)", np.asarray(diag["pipeline_coverage_raw"], dtype=np.float32)),
    ]

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            warped = _reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip Offroad layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Offroad proxy inputs (bbox 3035 @100 m)</summary>
  <p class="pl-hint">Rebuilt from OSM rail/pipeline layers, CORINE agri/industrial masks, LandScan population, and
  non-road composite — for comparison with <code>Offroad_Sourcearea.tif</code> on the national fine grid.
  Viridis + 2–98% stretch; zeros transparent.</p>
</details>"""
    return out_layers, leg


def _offroad_legend_block() -> str:
    return """
<details class="pl-details">
  <summary>Offroad (GNFR I)</summary>
  <p class="pl-hint">Context layers: rail and pipeline coverage, CORINE/OSM industry, population P_pop term, and raw non-road blend.</p>
</details>"""


def _build_waste_j_context_layers(
    root: Path,
    waste_cfg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
    sectors_data: dict | None,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Rebuild Waste / GNFR J solid-proxy ingredients on a cropped ref window, warp to WGS84 display grid.

    Returns (list of (layer_title, rgba uint8 HxWx4), HTML legend fragment).
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from Waste.j_waste_weights.io_utils import load_ref_profile
        from Waste.j_waste_weights.proxy_building import (
            OSM_WASTE_FAMILY_COVER,
            build_all_proxies,
            build_solid_waste_context_masks,
            build_wastewater_context_grids,
        )
    except ImportError as exc:
        print(f"Warning: Waste J context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    cfgp = waste_cfg_path if waste_cfg_path.is_absolute() else (root / waste_cfg_path)
    if not cfgp.is_file():
        print(f"Warning: Waste J config not found: {cfgp}", file=sys.stderr)
        return [], ""

    with cfgp.open(encoding="utf-8") as f:
        wcfg: dict = yaml.safe_load(f) or {}
    wcfg["_project_root"] = root
    wcfg["_config_path"] = cfgp

    paths = dict(wcfg.get("paths") or {})
    if sectors_data:
        main_paths = sectors_data.get("paths") or {}
        for key in ("cams_nc", "nuts_gpkg", "corine"):
            if key in main_paths:
                paths[key] = main_paths[key]
    wcfg["paths"] = paths

    try:
        ref = load_ref_profile(wcfg)
    except Exception as exc:
        print(f"Warning: Waste J ref grid failed: {exc}", file=sys.stderr)
        return [], ""

    mini = _intersect_ref_window_with_wgs84_bbox(ref, west, south, east, north)
    if mini is None:
        print(
            "Warning: map extent does not overlap the Waste J reference grid.",
            file=sys.stderr,
        )
        return [], ""

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])

    try:
        masks = build_solid_waste_context_masks(wcfg, mini)
        ww_ctx = build_wastewater_context_grids(wcfg, mini)
        proxies = build_all_proxies(wcfg, mini)
    except Exception as exc:
        print(f"Warning: Waste J context layer build failed: {exc}", file=sys.stderr)
        return [], ""

    def _title_for_mask_key(key: str) -> str:
        if key == "corine_clc_132":
            return "Waste J · CORINE CLC 132 (dump sites) · binary"
        if key == "corine_clc_121":
            return "Waste J · CORINE CLC 121 (mineral extraction sites) · binary"
        if key.startswith("osm_"):
            fam = key[4:]
            desc = OSM_WASTE_FAMILY_COVER.get(fam, fam)
            return f"Waste J · OSM waste_family={fam} ({desc}) · binary"
        return f"Waste J · {key}"

    ordered_keys: list[str] = []
    for k in ("corine_clc_132", "corine_clc_121"):
        if k in masks:
            ordered_keys.append(k)
    ordered_keys.extend(sorted(k for k in masks if k.startswith("osm_")))

    layers_3035: list[tuple[str, np.ndarray]] = []

    # Core Waste proxies (same module as the pipeline).
    for key in ("proxy_solid", "proxy_wastewater", "proxy_residual"):
        arr = proxies.get(key)
        if arr is not None:
            layers_3035.append((f"Waste J · {key}", np.asarray(arr, dtype=np.float32)))

    # Wastewater ingredients requested: agglomerations + plants, plus population/imperviousness & industrial CLC mask.
    for key in (
        "uwwtd_agglomerations",
        "uwwtd_treatment_plants",
        "industrial_clc_mask",
        "imperviousness",
        "population",
    ):
        arr = ww_ctx.get(key)
        if arr is not None:
            layers_3035.append((f"Waste J · wastewater · {key}", np.asarray(arr, dtype=np.float32)))
    if "imperv_valid_mask" in ww_ctx:
        layers_3035.append(
            ("Waste J · wastewater · imperv_valid_mask", np.asarray(ww_ctx["imperv_valid_mask"], dtype=np.float32))
        )
    for k in ordered_keys:
        layers_3035.append((_title_for_mask_key(k), np.asarray(masks[k], dtype=np.float32)))

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        try:
            warped = _reproject_array_to_wgs84_grid(
                grid,
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip Waste J layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Waste J proxy inputs (cropped ref window)</summary>
  <p class="pl-hint">Ingredients match <code>Waste/j_waste_weights/proxy_building.py</code> on the fine grid clipped to the map extent
  (YAML from <code>--waste-j-config</code>). Includes CORINE 132/121 + OSM <code>waste_family</code> masks, and wastewater inputs:
  UWWTD agglomerations + buffered treatment plants, population, imperviousness, and industrial CORINE mask. Viridis + 2–98% stretch; zeros transparent.</p>
</details>"""
    return out_layers, leg


def _fugitive_legend_block() -> str:
    return """
<details class="pl-details" open>
  <summary>Fugitive (GNFR D area)</summary>
  <p class="pl-hint">Output: <code>Fugitive_areasource.tif</code> under your <code>--outputs-root</code> — per-pixel
  shares per pollutant band,   normalized to sum 1 within each CAMS GNFR D parent cell. Toggle context layers for
  population, <code>P_pop</code>, and G1–G4 OSM / CLC / <code>P_g</code> ingredients.</p>
</details>"""


def _waste_j_legend_block() -> str:
    return """
<details class="pl-details" open>
  <summary>Waste (GNFR J area)</summary>
  <p class="pl-hint">Area output: <code>Waste_sourcearea.tif</code>; point CAMS stream: <code>Waste_pointarea.tif</code>
  (under <code>--outputs-root</code>) — per-pixel shares per pollutant band, normalized within each CAMS GNFR J
  area- or point-type cell mask. Toggle context layers for CORINE 132/121 and each OSM <code>waste_family</code>
  mask (same GPKG as the pipeline).</p>
</details>"""


def _solvents_legend_block() -> str:
    return """
<details class="pl-details" open>
  <summary>Solvents (GNFR E area)</summary>
  <p class="pl-hint">Output: <code>Solvents/outputs/E_solvents_areasource.tif</code> — per-pixel shares per pollutant band,
  normalized to sum 1 within each CAMS GNFR E parent cell. Context layers: CORINE groups from <code>corine_codes</code>,
  OSM per <code>solvent_family</code> from <code>data/Solvents/osm_solvent_layers.gpkg</code> when present (else aggregate
  landuse from PBF), plus road-length proxies. Use <code>--clip-bbox w,s,e,n</code> (WGS84) for a small map; default clip
  is Attica-sized if omitted with <code>--preset solvents</code>.</p>
</details>"""


def _build_waste_context_layers(
    root: Path,
    waste_cfg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_transform,
    dst_shape: tuple[int, int],
    *,
    resampling: str,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """
    Rebuild Waste proxy ingredients on a cropped ref window, warp to WGS84 display grid.

    Returns (list of (layer_title, rgba uint8 HxWx4), HTML legend fragment).
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from Waste.j_waste_weights.io_utils import load_ref_profile, warp_raster_to_ref
        from Waste.j_waste_weights.proxy_building import (
            build_all_proxies,
            build_solid_waste_context_masks,
            build_wastewater_context_grids,
        )
        from rasterio.enums import Resampling
    except ImportError as exc:
        print(f"Warning: Waste context layers skipped: {exc}", file=sys.stderr)
        return [], ""

    cfgp = waste_cfg_path if waste_cfg_path.is_absolute() else (root / waste_cfg_path)
    if not cfgp.is_file():
        print(f"Warning: waste config not found: {cfgp}", file=sys.stderr)
        return [], ""

    try:
        cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Warning: waste config read failed: {exc}", file=sys.stderr)
        return [], ""
    if not isinstance(cfg, dict):
        print("Warning: waste config is not a dict; context layers skipped.", file=sys.stderr)
        return [], ""
    cfg["_project_root"] = root
    cfg["_config_path"] = cfgp

    try:
        ref = load_ref_profile(cfg)
    except Exception as exc:
        print(f"Warning: Waste ref grid failed: {exc}", file=sys.stderr)
        return [], ""

    mini = _intersect_ref_window_with_wgs84_bbox(ref, west, south, east, north)
    if mini is None:
        print("Warning: map extent does not overlap the Waste reference grid.", file=sys.stderr)
        return [], ""

    crs_str = str(mini["crs"])
    transform = mini["transform"]
    gh, gw = int(dst_shape[0]), int(dst_shape[1])

    layers_3035: list[tuple[str, np.ndarray]] = []
    try:
        # Core proxies (solid / wastewater / residual) on the same cropped ref grid.
        proxies = build_all_proxies(cfg, mini)
        layers_3035.extend(
            [
                ("Waste · proxy_solid", np.asarray(proxies.get("proxy_solid"), dtype=np.float32)),
                ("Waste · proxy_wastewater", np.asarray(proxies.get("proxy_wastewater"), dtype=np.float32)),
                ("Waste · proxy_residual", np.asarray(proxies.get("proxy_residual"), dtype=np.float32)),
            ]
        )

        # Solid-family masks: CORINE 132/121 and OSM waste-family layers (if configured).
        solid_masks = build_solid_waste_context_masks(cfg, mini)
        for k in sorted(solid_masks.keys()):
            layers_3035.append((f"Waste · solid mask · {k}", np.asarray(solid_masks[k], dtype=np.float32)))

        # Wastewater intermediate grids: agglomerations, plants, population, imperviousness, industrial CLC mask.
        ww = build_wastewater_context_grids(cfg, mini)
        for k in (
            "uwwtd_agglomerations",
            "uwwtd_treatment_plants",
            "industrial_clc_mask",
            "imperviousness",
            "population",
        ):
            if k in ww:
                layers_3035.append((f"Waste · wastewater · {k}", np.asarray(ww[k], dtype=np.float32)))

        # Optional: show where imperviousness had valid coverage.
        if "imperv_valid_mask" in ww:
            layers_3035.append(
                ("Waste · imperviousness valid mask", np.asarray(ww["imperv_valid_mask"], dtype=np.float32))
            )

        # Also show the CORINE raster codes on the cropped window (nearest).
        cor_path = Path(mini["corine_path"])
        if cor_path.is_file():
            clc = warp_raster_to_ref(
                cor_path,
                mini,
                band=1,
                resampling=Resampling.nearest,
                src_nodata=None,
                dst_nodata=np.nan,
            )
            layers_3035.append(("Waste · CORINE CLC codes (nearest)", np.asarray(clc, dtype=np.float32)))
    except Exception as exc:
        print(f"Warning: Waste proxy rebuild failed: {exc}", file=sys.stderr)
        return [], ""

    out_layers: list[tuple[str, np.ndarray]] = []
    for title, grid in layers_3035:
        if grid is None:
            continue
        try:
            warped = _reproject_array_to_wgs84_grid(
                np.asarray(grid, dtype=np.float32),
                transform,
                crs_str,
                dst_transform,
                (gh, gw),
                resampling=resampling,
            )
        except Exception as exc:
            print(f"Warning: skip Waste layer {title!r}: {exc}", file=sys.stderr)
            continue
        rgba = _scalar_to_rgba(
            warped,
            colour_mode="percentile",
            cmap_name="viridis",
            hide_zero=True,
            nodata_val=None,
        )
        if np.any(rgba[..., 3] > 0):
            out_layers.append((title, rgba))

    leg = """
<details class="pl-details">
  <summary>Waste proxy inputs (cropped ref window)</summary>
  <p class="pl-hint">Layers follow <code>Waste/j_waste_weights/proxy_building.py</code> on the fine grid clipped to the map extent
  (YAML from <code>--waste-config</code>). Includes CORINE/OSM solid masks, UWWTD agglomerations + treatment-plant buffers,
  and population/imperviousness ingredients for wastewater/residual proxies. Viridis + 2–98% stretch; zeros transparent.</p>
</details>"""
    return out_layers, leg


def _waste_legend_block() -> str:
    return """
<details class="pl-details" open>
  <summary>Waste (GNFR J)</summary>
  <p class="pl-hint">Pass a Waste weights GeoTIFF via <code>--tif</code> (e.g. <code>Waste_sourcearea.tif</code> or <code>Waste_pointarea.tif</code>).
  Toggle the proxy-input layers (CORINE/OSM solid masks, UWWTD agglomerations / plants, population / imperviousness) to debug the proxy build.</p>
</details>"""


def _build_residential_context_layers(
    root: Path,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    dst_t,
    gh: int,
    gw: int,
    res_cfg: dict,
    sectors_data: dict,
    resampling: str,
) -> tuple[
    np.ndarray | None,
    list[tuple[str, np.ndarray]],
    np.ndarray | None,
    list[dict],
]:
    """
    CORINE L1 (codes 1–3), Hotmaps, LandScan.
    Returns (corine_rgba, hot_layers, pop_rgba, hot_cbar_specs).
    """
    reproject_one = _load_reproject_helper()
    paths_res = res_cfg.get("paths") or {}
    paths_sec = sectors_data.get("paths") or {}
    cor_p = paths_res.get("corine")
    cor_path = None
    if cor_p:
        cor_path = _resolve_under_root(root, cor_p)
    if cor_path is None or not cor_path.is_file():
        try:
            cor_path = first_existing_corine(root, paths_sec.get("corine"))
        except FileNotFoundError:
            cor_path = None
    rc = res_cfg.get("corine") or {}
    sc = sectors_data.get("corine") or {}
    if "band" in rc:
        cor_band = int(rc["band"])
    elif "band" in sc:
        cor_band = int(sc["band"])
    else:
        cor_band = 1
    if cor_band < 1:
        cor_band = 1

    cor_rgba: np.ndarray | None = None
    if cor_path is not None and cor_path.is_file():
        try:
            clc = _corine_class_grid_on_display_grid(
                cor_path, cor_band, dst_t, gh, gw, reproject_one
            )
            cor_rgba = _corine_residential_l1_overlay_rgba(clc)
            if not np.any(cor_rgba[..., 3] > 0):
                u = (
                    np.unique(clc[(clc >= 1) & (clc <= 44)])
                    if np.any((clc >= 1) & (clc <= 44))
                    else np.array([], dtype=np.int32)
                )
                print(
                    "[residential viz] No pixels with CORINE L1 codes 1, 2, or 3 on this map grid. "
                    f"Unique L1 codes 1–44 in view (sample): {u[:24]}. "
                    "Check CORINE raster legend (1–44) vs downscaling reference extent.",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[residential viz] CORINE overlay failed: {exc}", file=sys.stderr)
            cor_rgba = None

    def _hm_group(key: str) -> str:
        if key.startswith("heat"):
            return "heat"
        if key.startswith("gfa"):
            return "gfa"
        return key

    def _same_raster_pixels(a: np.ndarray, b: np.ndarray) -> bool:
        if a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True))

    hm = paths_res.get("hotmaps") or {}
    hm_specs = (
        ("Hotmaps heat demand (res.)", "heat_res"),
        ("Hotmaps heat demand (non-res.)", "heat_nonres"),
        ("Hotmaps GFA (res.)", "gfa_res"),
        ("Hotmaps GFA (non-res.)", "gfa_nonres"),
    )

    collected: list[tuple[str, str, Path, np.ndarray]] = []
    for title, key in hm_specs:
        rel = hm.get(key)
        hp = _resolve_under_root(root, rel)
        if hp is None or not hp.is_file():
            continue
        try:
            arr = reproject_one(
                hp,
                dst_t,
                (gh, gw),
                resampling=resampling,
                band=1,
            )
        except Exception as exc:
            print(f"Warning: skip Hotmaps {key}: {exc}", file=sys.stderr)
            continue
        arr64 = np.asarray(arr, dtype=np.float64)
        collected.append((title, key, hp, arr64))

    pr_heat = next((hp for _t, k, hp, _a in collected if k == "heat_res"), None)
    pn_heat = next((hp for _t, k, hp, _a in collected if k == "heat_nonres"), None)
    if (
        pr_heat is not None
        and pn_heat is not None
        and pr_heat.resolve() == pn_heat.resolve()
    ):
        print(
            "[residential viz] heat_res and heat_nonres use the same file path; "
            "fix paths in residential config or re-download distinct Hotmaps rasters.",
            file=sys.stderr,
        )
    pr_gfa = next((hp for _t, k, hp, _a in collected if k == "gfa_res"), None)
    pn_gfa = next((hp for _t, k, hp, _a in collected if k == "gfa_nonres"), None)
    if (
        pr_gfa is not None
        and pn_gfa is not None
        and pr_gfa.resolve() == pn_gfa.resolve()
    ):
        print(
            "[residential viz] gfa_res and gfa_nonres use the same file path.",
            file=sys.stderr,
        )

    grouped: dict[str, list[tuple[str, str, Path, np.ndarray]]] = {}
    for item in collected:
        grouped.setdefault(_hm_group(item[1]), []).append(item)

    hot_layers: list[tuple[str, np.ndarray]] = []
    hot_cbar_specs: list[dict] = []
    cmap_by_group = {"heat": "viridis", "gfa": "plasma"}
    for g_name, items in grouped.items():
        pruned: list[tuple[str, str, Path, np.ndarray]] = []
        for title, key, hp, arr64 in items:
            if pruned and _same_raster_pixels(pruned[-1][3], arr64):
                print(
                    f"[residential viz] Skipping {title!r}: pixel values match {pruned[-1][0]!r} "
                    f"(same or duplicated Hotmaps data). File: {hp}",
                    file=sys.stderr,
                )
                continue
            pruned.append((title, key, hp, arr64))
        if not pruned:
            continue
        cmap_g = cmap_by_group.get(g_name, "viridis")
        arrs = [x[3] for x in pruned]
        glob_bounds = _wp_global_log_bounds(arrs)
        if glob_bounds is not None:
            hot_cbar_specs.append(
                {
                    "group": g_name,
                    "cmap": cmap_g,
                    "log_vmin": float(glob_bounds[0]),
                    "log_vmax": float(glob_bounds[1]),
                    "layers": [x[0] for x in pruned],
                }
            )
        for title, _key, _hp, arr64 in pruned:
            rgba_h = _wp_importance_rgba(
                arr64,
                cmap_name=cmap_g,
                log_vmin=glob_bounds[0] if glob_bounds else None,
                log_vmax=glob_bounds[1] if glob_bounds else None,
                alpha=220,
            )
            if np.any(rgba_h[..., 3] > 0):
                hot_layers.append((title, rgba_h))

    pop_rgba: np.ndarray | None = None
    ls_rel = paths_sec.get("landscan")
    ls_path = _resolve_under_root(root, ls_rel)
    if ls_path is not None and ls_path.is_file():
        try:
            pop = reproject_one(
                ls_path,
                dst_t,
                (gh, gw),
                resampling=resampling,
                band=1,
            )
            pop64 = np.asarray(pop, dtype=np.float64)
            pop_rgba = _wp_importance_rgba(
                pop64,
                cmap_name=WP_IMPORTANCE_CMAP,
                log_vmin=None,
                log_vmax=None,
                alpha=200,
            )
            if not np.any(pop_rgba[..., 3] > 0):
                pop_rgba = None
        except Exception as exc:
            print(f"Warning: LandScan overlay skipped: {exc}", file=sys.stderr)

    return cor_rgba, hot_layers, pop_rgba, hot_cbar_specs


def _residential_corine_l1_legend_rows_html() -> str:
    rows = []
    for code in sorted(RESIDENTIAL_CORINE_L1.keys()):
        name, col = RESIDENTIAL_CORINE_L1[code]
        rows.append(
            f'<div class="pl-clc-row">'
            f'<span class="pl-swatch" style="background:{col};"></span>'
            f"<span><b>{int(code)}</b> {html.escape(name)} "
            f'<code style="font-size:8px">{html.escape(col)}</code></span></div>'
        )
    return "".join(rows)


def _residential_legend_html(
    *,
    show_corine_l1: bool,
    hotmap_titles: list[str],
    pop_ok: bool,
    hot_cbar_specs: list[dict] | None = None,
    spread_legend_fragment: str = "",
) -> str:
    cor_block = ""
    if show_corine_l1:
        cor_body = _residential_corine_l1_legend_rows_html()
        cor_block = f"""
<details class="pl-details" open>
  <summary>CORINE Level 1 (raster codes 1–3)</summary>
  <div class="pl-clc-grid">{cor_body}</div>
  <p class="pl-hint">Other L1 classes are not drawn. Warp matches the proxy grid (nearest).</p>
</details>"""
    hm_block = ""
    cbar_blocks = ""
    specs = hot_cbar_specs or []
    for spec in specs:
        cmap_name = str(spec.get("cmap", "viridis"))
        lo = float(spec.get("log_vmin", 0.0))
        hi = float(spec.get("log_vmax", 0.0))
        grp = html.escape(str(spec.get("group", "hotmaps")))
        layers = spec.get("layers") or []
        layer_txt = html.escape(", ".join(str(x) for x in layers))
        grad_hex = _sample_cmap_hex(cmap_name, n=11)
        grad_css = _css_linear_gradient_stops(grad_hex)
        cbar_blocks += f"""
<div class="pl-hm-cbar">
  <p class="pl-meta"><b>{grp}</b> · log10 · <code>{html.escape(cmap_name)}</code></p>
  <div class="pl-grad" style="background:{grad_css};"></div>
  <div class="pl-grad-lbl"><span>{lo:.3f}</span><span>{hi:.3f}</span></div>
  <p class="pl-hint">{layer_txt}</p>
</div>"""
    if hotmap_titles or cbar_blocks:
        lst = ", ".join(html.escape(t) for t in hotmap_titles)
        hm_block = f"""
<details class="pl-details">
  <summary>Hotmaps (log10)</summary>
  <p class="pl-hint">Heat: viridis (shared within heat pair). GFA: plasma. Layers: {lst}</p>
  {cbar_blocks}
</details>"""
    pop_block = ""
    if pop_ok:
        pop_block = """
<details class="pl-details">
  <summary>LandScan population</summary>
  <p class="pl-hint">log10(count), YlOrRd (from sectors.json <code>paths.landscan</code>).</p>
</details>"""
    return cor_block + hm_block + pop_block + spread_legend_fragment


def _css_linear_gradient_stops(hex_colours: list[str]) -> str:
    n = len(hex_colours)
    if n == 0:
        return "#ccc"
    if n == 1:
        return hex_colours[0]
    parts = [f"{h} {100.0 * i / (n - 1):.2f}%" for i, h in enumerate(hex_colours)]
    return "linear-gradient(to right," + ",".join(parts) + ")"


def _corine_legend_compact_rows_html(ag_codes: tuple[int, ...]) -> str:
    rows = []
    for c in ag_codes:
        col = CLC_AG_COLORS.get(int(c), "#888")
        name = html.escape(CLC_AG_NAMES.get(int(c), f"CLC {c}"))
        rows.append(
            f'<div class="pl-clc-row">'
            f'<span class="pl-swatch" style="background:{col};"></span>'
            f"<span><b>{c}</b> {name}</span></div>"
        )
    return "".join(rows)


def _combined_map_legend_html(
    *,
    tif_name: str,
    band: int,
    colour_mode: str,
    cmap_name: str,
    mode_caption_long: str,
    mode_caption_short: str,
    gradient_hex: list[str],
    ag_codes: tuple[int, ...] | None,
    wp_pollutant_names: list[str] | None = None,
    wp_log_range: tuple[float, float] | None = None,
    wp_cmap_legend: str = WP_IMPORTANCE_CMAP,
    wp_viz: str = "nuts",
    extra_legend_html: str = "",
    active_band_html: str = "",
) -> str:
    safe_tif = html.escape(tif_name)
    safe_long = html.escape(mode_caption_long)
    safe_short = html.escape(mode_caption_short)
    grad = _css_linear_gradient_stops(gradient_hex)
    corine_block = ""
    if ag_codes:
        corine_body = _corine_legend_compact_rows_html(ag_codes)
        corine_block = f"""
<details class="pl-details">
  <summary>CORINE classes (high-contrast)</summary>
  <div class="pl-clc-grid">{corine_body}</div>
  <p class="pl-hint">Areas with <b>no fill</b> are not ag CLC 12–22 (basemap visible). Hairlines separate two different ag classes.</p>
</details>"""
    wp_leg = ""
    if wp_pollutant_names:
        lo, hi = wp_log_range if wp_log_range else (0.0, 0.0)
        rng_txt = f"{lo:.2f} … {hi:.2f}" if wp_log_range else "n/a"
        pol_list = ", ".join(html.escape(p) for p in wp_pollutant_names)
        if wp_viz == "nuts":
            mode_txt = (
                "Each NUTS2 polygon is a <b>single colour</b>: sum of <code>S_p</code> over agricultural CLC rows in "
                "<code>weights_long.csv</code> (stronger contrast between regions than pixel-wise <code>w_p</code>). "
                "Colour scale uses log10; for <code>nuts</code> mode the range is clipped at the 5th–95th percentile "
                "across regions to avoid one outlier washing the map."
            )
        else:
            mode_txt = (
                "Per-pixel <code>w_p</code> for each NUTS2×CLC combination (CORINE class on the ground). "
                "Same log10 min–max across all selected pollutants."
            )
        wp_leg = f"""
<details class="pl-details">
  <summary>weights_long ({len(wp_pollutant_names)} pollutant(s))</summary>
  <p class="pl-hint">{mode_txt}</p>
  <p class="pl-hint">Palette <code>{html.escape(wp_cmap_legend)}</code>, shared log10 axis <b>{html.escape(rng_txt)}</b>. Colorbar on the map.</p>
  <p class="pl-meta">{pol_list}</p>
</details>"""
    active_blk = active_band_html if active_band_html.strip() else ""
    return f"""
<div class="proxy-map-legend">
  {active_blk}
  <details open class="pl-details">
    <summary>Proxy: {safe_short} · {html.escape(cmap_name)}</summary>
    <p class="pl-meta"><code>{safe_tif}</code> b{int(band)} · {html.escape(colour_mode)}</p>
    <p class="pl-meta pl-long">{safe_long}</p>
    <div class="pl-grad" style="background:{grad};"></div>
    <div class="pl-grad-lbl"><span>0</span><span>1</span></div>
  </details>
  {corine_block}
  {wp_leg}
  {extra_legend_html}
</div>
<style>
.proxy-map-legend {{
  position: fixed !important;
  bottom: 10px !important;
  left: 10px !important;
  width: min(272px, 46vw) !important;
  max-height: min(320px, 42vh) !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  z-index: 650 !important;
  background: rgba(255,255,255,0.93) !important;
  border: 1px solid #999 !important;
  border-radius: 6px !important;
  padding: 6px 8px !important;
  box-shadow: 0 1px 5px rgba(0,0,0,0.18) !important;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}
.proxy-map-legend .pl-details {{ margin: 0; padding: 2px 0; border-top: 1px solid #ddd; }}
.proxy-map-legend .pl-details:first-of-type {{ border-top: none; padding-top: 0; }}
.proxy-map-legend summary {{
  cursor: pointer;
  font-weight: 600;
  font-size: 11px;
  list-style: none;
}}
.proxy-map-legend summary::-webkit-details-marker {{ display: none; }}
.proxy-map-legend .pl-meta {{ margin: 4px 0 0; font-size: 9px; color: #333; line-height: 1.25; }}
.proxy-map-legend .pl-long {{ color: #555; font-size: 9px; }}
.proxy-map-legend .pl-grad {{
  height: 11px;
  border-radius: 2px;
  border: 1px solid #666;
  margin-top: 6px;
}}
.proxy-map-legend .pl-grad-lbl {{
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: #444;
  margin-top: 1px;
}}
.proxy-map-legend .pl-clc-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2px 6px;
  margin-top: 4px;
}}
.proxy-map-legend .pl-clc-row {{
  display: flex;
  align-items: flex-start;
  gap: 4px;
  font-size: 9px;
  line-height: 1.2;
}}
.proxy-map-legend .pl-swatch {{
  width: 10px;
  height: 10px;
  flex-shrink: 0;
  border: 1px solid #666;
  margin-top: 1px;
}}
.proxy-map-legend .pl-hint {{
  margin: 6px 0 0;
  font-size: 9px;
  color: #555;
  line-height: 1.3;
}}
.proxy-map-legend .pl-hm-cbar {{
  margin-top: 8px;
  padding-top: 4px;
  border-top: 1px solid #e0e0e0;
}}
</style>
"""


PRESET_FILENAMES = {
    "agriculture": "Agriculture_sourcearea.tif",
    "fugitive_area": "Fugitive_areasource.tif",
    "offroad": "Offroad_Sourcearea.tif",
    "public_power_area": "PublicPower_sourcearea.tif",
    "public_power_point": "PublicPower_sourcepoint.tif",
    "residential": "Residential_sourcearea.tif",
    "shipping": "Shipping_areasource.tif",
    "solvents": "Solvents/outputs/E_solvents_areasource.tif",
    "waste_j": "Waste_sourcearea.tif",
}

PRESET_OUTPUT_ROOT_DEFAULT = Path("SourceProxies") / "outputs" / "EL"
RESIDENTIAL_LEGACY_DIR = Path("Residential") / "outputs" / "downscaled"
DEFAULT_RESIDENTIAL_CONFIG = Path("Residential") / "config" / "residential_cams_downscale.config.json"

PRESET_TO_CAMS_SECTOR = {
    "agriculture": "ag_kl",
    "fugitive_area": "gnfr_d",
    "offroad": "gnfr_i",
    "public_power_area": "public_power_a",
    "public_power_point": "public_power_a",
    "residential": "gnfr_c",
    "shipping": "gnfr_g",
    "solvents": "gnfr_e",
    "waste_j": "gnfr_j",
}


def _tif_suggests_agriculture_overlay(tif_path: Path) -> bool:
    """Enable CORINE/w_p when user passes --tif without --preset."""
    n = tif_path.name.lower().replace("-", "_")
    ag = str(PRESET_FILENAMES["agriculture"]).lower().replace("-", "_")
    return n == ag or "agriculture_sourcearea" in n


def _tif_suggests_solvents(tif_path: Path) -> bool:
    n = tif_path.name.lower().replace("-", "_")
    return "e_solvents_areasource" in n or "solvents_areasource" in n


def _tif_suggests_fugitive(tif_path: Path) -> bool:
    n = tif_path.name.lower().replace("-", "_")
    return n == "fugitive_areasource.tif" or "fugitive_areasource" in n


def _tif_suggests_waste_j(tif_path: Path) -> bool:
    n = tif_path.name.lower().replace("-", "_")
    return (
        "waste_sourcearea" in n
        or "cams_j_waste_within_cell_weights" in n
        or "j_waste_within_cell" in n
    )


def _tif_suggests_shipping(tif_path: Path) -> bool:
    n = tif_path.name.lower().replace("-", "_")
    return n == "shipping_areasource.tif" or "shipping_areasource" in n


def _tif_suggests_offroad(tif_path: Path) -> bool:
    n = tif_path.name.lower().replace("-", "_")
    return n == "offroad_sourcearea.tif" or "offroad_sourcearea" in n


def _tif_suggests_residential_overlay(tif_path: Path) -> bool:
    """Enable residential context layers for GNFR C proxy rasters."""
    stem = tif_path.stem.lower()
    if stem == "residential_sourcearea":
        return True
    parts_lower = {p.lower() for p in tif_path.parts}
    if "downscaled" not in parts_lower:
        return False
    return stem.startswith("weights_") or stem.startswith("emissions_")


def _effective_cams_sector(args: argparse.Namespace) -> str | None:
    if str(args.cams_sector) != "auto":
        return str(args.cams_sector)
    if args.preset:
        return PRESET_TO_CAMS_SECTOR.get(str(args.preset))
    return None


def _effective_outputs_root(root: Path, args: argparse.Namespace) -> Path:
    if args.outputs_root is not None:
        p = args.outputs_root
        return p if p.is_absolute() else (root / p)
    return root / PRESET_OUTPUT_ROOT_DEFAULT


def _resolve_tif_path(
    root: Path,
    args: argparse.Namespace,
) -> Path:
    if args.tif is not None:
        p = args.tif if args.tif.is_absolute() else root / args.tif
        return p
    if args.preset is None:
        raise SystemExit("Provide --tif or --preset (with --outputs-root).")
    sub = PRESET_FILENAMES.get(str(args.preset))
    if sub is None:
        raise SystemExit(
            f"Unknown --preset {args.preset!r}. Choose from: {', '.join(sorted(PRESET_FILENAMES))}."
        )
    out_root = _effective_outputs_root(root, args)
    if str(args.preset) == "residential":
        primary = out_root / "Residential_sourcearea.tif"
        if primary.is_file():
            return primary
        pol = str(args.residential_pollutant).strip().lower().replace("-", "_")
        legacy = root / RESIDENTIAL_LEGACY_DIR
        for stem in (f"weights_{pol}", f"emissions_{pol}"):
            cand = legacy / f"{stem}.tif"
            if cand.is_file():
                return cand
        return primary
    if str(args.preset) == "solvents":
        p = Path(sub)
        return p if p.is_absolute() else (root / p)
    if str(args.preset) == "shipping":
        p = Path(sub)
        return p if p.is_absolute() else (root / p)
    if str(args.preset) == "waste_j":
        wcfg_path = args.waste_j_config
        if wcfg_path is None:
            wcfg_path = Path("Waste/j_waste_weights/config.yaml")
        if not wcfg_path.is_absolute():
            wcfg_path = root / wcfg_path
        with wcfg_path.open(encoding="utf-8") as wf:
            wcfg = yaml.safe_load(wf) or {}
        out_o = wcfg.get("output") or {}
        wt = str(
            out_o.get("weights_tif_area")
            or out_o.get("weights_tif")
            or "Waste_sourcearea.tif"
        )
        return out_root / wt
    return out_root / sub


def _sample_cmap_hex(cmap_name: str, n: int = 11) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    cmap = colormaps[cmap_name]
    xs = np.linspace(0.0, 1.0, n)
    rgba = cmap(xs)
    out: list[str] = []
    for r, g, b, a in rgba:
        out.append(
            f"#{int(255 * r):02x}{int(255 * g):02x}{int(255 * b):02x}"
        )
    return out


def _resolve_display_pollutant(sidecar: dict, band_1based: int) -> str | None:
    bands = sidecar.get("bands")
    if not isinstance(bands, list):
        return None
    for row in bands:
        if int(row.get("index", -1)) == int(band_1based):
            p = str(row.get("pollutant", "")).strip().upper()
            return p if p else None
    return None


def _cross_pollutant_weight_spread_rgba(
    tif_path: Path,
    dst_t,
    gh: int,
    gw: int,
    *,
    resampling: str,
    reproject_one,
) -> np.ndarray | None:
    import rasterio

    with rasterio.open(tif_path) as src:
        n = int(src.count)
    if n < 2:
        return None
    stack: list[np.ndarray] = []
    for b in range(1, n + 1):
        stack.append(
            np.asarray(
                reproject_one(
                    tif_path,
                    dst_t,
                    (gh, gw),
                    resampling=resampling,
                    band=b,
                ),
                dtype=np.float64,
            )
        )
    S = np.stack(stack, axis=0)
    finite = np.isfinite(S)
    n_ok = np.sum(finite, axis=0)
    spread = np.full((gh, gw), np.nan, dtype=np.float64)
    m = n_ok >= 2
    if np.any(m):
        spread[m] = np.nanstd(S[:, m], axis=0)
    rgba = _scalar_to_rgba(
        spread,
        colour_mode="percentile",
        cmap_name="cividis",
        hide_zero=True,
        nodata_val=None,
    )
    if not np.any(rgba[..., 3] > 0):
        return None
    return rgba


def _scalar_to_rgba(
    z: np.ndarray,
    *,
    colour_mode: str,
    cmap_name: str,
    hide_zero: bool,
    nodata_val: float | None,
    z_precomputed_01: np.ndarray | None = None,
    valid_precomputed: np.ndarray | None = None,
) -> np.ndarray:
    """Return uint8 (H, W, 4) RGBA for Folium ImageOverlay."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if z_precomputed_01 is not None:
        t_full = np.asarray(z_precomputed_01, dtype=np.float64)
        valid = np.asarray(valid_precomputed, dtype=bool)
        if hide_zero:
            valid = valid & (z > 0)
        if nodata_val is not None:
            valid = valid & np.isfinite(z) & (z != float(nodata_val))
    else:
        finite = np.isfinite(z)
        if nodata_val is not None:
            finite = finite & (z != float(nodata_val))
        if hide_zero:
            valid = finite & (z > 0)
        else:
            valid = finite
        if not np.any(valid):
            return rgba
        vals = z[valid]

        if colour_mode == "log":
            vals = np.log10(np.maximum(vals, 1e-18))
        elif colour_mode == "percentile":
            lo = float(np.percentile(vals, 2.0))
            hi = float(np.percentile(vals, 98.0))
            if lo >= hi:
                hi = lo + 1e-9
            vals = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)

        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo >= hi:
            hi = lo + 1e-6
        t = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        cmap = colormaps[cmap_name]
        c = cmap(t)
        rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
        rgba[valid, 3] = 255
        return rgba

    if not np.any(valid):
        return rgba

    cmap = colormaps[cmap_name]
    t = np.clip(t_full[valid], 0.0, 1.0)
    c = cmap(t)
    rgba[valid, 0] = (np.clip(c[:, 0], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 1] = (np.clip(c[:, 1], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 2] = (np.clip(c[:, 2], 0.0, 1.0) * 255).astype(np.uint8)
    rgba[valid, 3] = 255
    return rgba


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="Map a proxy weight GeoTIFF (warped to WGS84) with Folium.",
    )
    ap.add_argument(
        "--tif",
        type=Path,
        default=None,
        help="Path to GeoTIFF (relative paths from project root)",
    )
    ap.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Directory for preset outputs (default: SourceProxies/outputs/EL)",
    )
    ap.add_argument(
        "--preset",
        choices=sorted(PRESET_FILENAMES.keys()),
        default=None,
        help=(
            "Shorthand under --outputs-root (or residential downscaled dir). "
            "agriculture: CORINE/w_p overlays; residential: urban CORINE + Hotmaps + LandScan; "
            "solvents: E_solvents_areasource.tif + proxy-input layers on a clipped window; "
            "fugitive_area: Fugitive_areasource.tif + population / OSM / CLC / P_g context layers; "
            "offroad: Offroad_Sourcearea.tif + rail/pipeline CORINE/population context layers; "
            "shipping: Shipping_areasource.tif + EMODnet / OSM / CLC port context layers; "
            "waste_j: Waste_sourcearea.tif (+ Waste_pointarea.tif) + CORINE 121/132 and OSM waste_family masks."
        ),
    )
    ap.add_argument("--band", type=int, default=1, help="1-based band index")
    ap.add_argument(
        "--colour-mode",
        choices=("global", "log", "percentile", "per_cams_cell"),
        default="global",
        help="per_cams_cell = 0-1 contrast within each CAMS area cell (needs --cams-nc or config)",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="plasma",
        help="Matplotlib colormap (default: plasma; good on light basemaps)",
    )
    ap.add_argument(
        "--cams-nc",
        type=Path,
        default=None,
        help="CAMS NetCDF (for per_cams_cell); default from SourceProxies/config/sectors.json",
    )
    ap.add_argument(
        "--cams-iso3",
        type=str,
        default="GRC",
        help="CAMS country_id ISO3 for per_cams_cell masks",
    )
    ap.add_argument(
        "--cams-sector",
        choices=(
            "auto",
            "ag_kl",
            "public_power_a",
            "gnfr_c",
            "gnfr_d",
            "gnfr_e",
            "gnfr_g",
            "gnfr_i",
            "gnfr_j",
        ),
        default="auto",
        help="Which CAMS area sources define cells (auto from --preset; gnfr_c / d / e / g / i / j = GNFR C / D / E / G / I / J area)",
    )
    ap.add_argument(
        "--hide-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat zero as transparent (default: true). Use --no-hide-zero to show zeros.",
    )
    ap.add_argument("--pad-deg", type=float, default=0.02, help="Pad bounds in degrees")
    ap.add_argument(
        "--max-width",
        type=int,
        default=1800,
        help="Overlay width in pixels (higher = finer CORINE / w_p detail)",
    )
    ap.add_argument(
        "--max-height",
        type=int,
        default=1500,
        help="Overlay height in pixels",
    )
    ap.add_argument("--opacity", type=float, default=0.85, help="ImageOverlay opacity")
    ap.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output HTML (default: <tif stem>_proxy_map.html next to the tif)",
    )
    ap.add_argument(
        "--resampling",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="Warp resampling to WGS84 display grid",
    )
    ap.add_argument(
        "--no-cams-grid",
        action="store_true",
        help="With per_cams_cell: do not draw CAMS area cell outlines",
    )
    ap.add_argument(
        "--no-ag-overlays",
        action="store_true",
        help="Do not add CORINE ag-class and w_p importance rasters",
    )
    ap.add_argument(
        "--ag-overlays",
        action="store_true",
        help="Force CORINE + w_p overlays (needs sectors.json paths)",
    )
    ap.add_argument(
        "--pollutant",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Pollutant(s) for w_p overlay; repeat flag for several, e.g. --pollutant NH3 --pollutant CH4. "
            "If omitted, uses every entry in agriculture sector pollutants in sectors.json."
        ),
    )
    ap.add_argument(
        "--wp-viz",
        choices=("nuts", "pixels"),
        default="nuts",
        help=(
            "weights_long thermal overlay: 'nuts' = log10(sum S_p) per NUTS2 over ag CLCs (regional contrast); "
            "'pixels' = per-pixel w_p for each NUTS×CLC cell (finer but often flat)."
        ),
    )
    ap.add_argument(
        "--residential-pollutant",
        type=str,
        default="nh3",
        metavar="NAME",
        help=(
            "Legacy fallback pollutant file name when Residential_sourcearea.tif is missing "
            "(default: nh3)"
        ),
    )
    ap.add_argument(
        "--residential-config",
        type=Path,
        default=None,
        help="Residential downscale JSON (default: Residential/config/residential_cams_downscale.config.json)",
    )
    ap.add_argument(
        "--residential-overlays",
        action="store_true",
        help="Force CORINE/Hotmaps/LandScan context layers (with custom --tif)",
    )
    ap.add_argument(
        "--no-residential-overlays",
        action="store_true",
        help="Disable residential context layers when using --preset residential",
    )
    ap.add_argument(
        "--no-cross-pollutant-spread",
        action="store_true",
        help="Do not add std-dev-across-bands overlay (residential / solvents multi-band proxy)",
    )
    ap.add_argument(
        "--clip-bbox",
        type=str,
        default=None,
        metavar="W,S,E,N",
        help=(
            "WGS84 map extent west,south,east,north (commas or spaces). "
            "Overrides bounds derived from the GeoTIFF (useful for solvents preview)."
        ),
    )
    ap.add_argument(
        "--solvents-config",
        type=Path,
        default=None,
        help="Solvents JSON config (default: Solvents/config/solvents.defaults.json); used for context layers with --preset solvents.",
    )
    ap.add_argument(
        "--solvents-skip-osm",
        action="store_true",
        help="Set SOLVENTS_SKIP_OSM=1 while building solvent context layers (population/CORINE only; fast).",
    )
    ap.add_argument(
        "--fugitive-config",
        type=Path,
        default=None,
        help="Fugitive YAML (default: PROXY/config/fugitive/area_source.yaml); used for context layers with --preset fugitive_area or Fugitive_areasource.tif.",
    )
    ap.add_argument(
        "--offroad-config",
        type=Path,
        default=None,
        help="Offroad YAML (default: Offroad/config/offroad_area.yaml); used for context layers with --preset offroad or Offroad_Sourcearea.tif.",
    )
    ap.add_argument(
        "--waste-j-config",
        type=Path,
        default=None,
        dest="waste_j_config",
        help="Waste J YAML (default: Waste/j_waste_weights/config.yaml); used for context layers with --preset waste_j.",
    )
    args = ap.parse_args()

    tif_path = _resolve_tif_path(root, args)
    if not tif_path.is_file():
        raise SystemExit(f"GeoTIFF not found: {tif_path}")

    try:
        import folium
        import rasterio
        from branca.colormap import LinearColormap
        from rasterio.transform import from_bounds, xy as transform_xy
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise SystemExit("Need folium, branca, rasterio, matplotlib, numpy.") from exc

    reproject_one = _load_reproject_helper()
    sidecar_proxy = _load_json_if_exists(tif_path.with_suffix(".json"))

    with rasterio.open(tif_path) as src:
        n_bands_file = int(src.count)
        if args.band < 1 or args.band > n_bands_file:
            raise SystemExit(
                f"{tif_path.name}: band {args.band} invalid (file has {n_bands_file})."
            )
        if src.crs is None:
            raise SystemExit(f"Raster has no CRS: {tif_path}")
        nodata = src.nodata
        w, s, e, n = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )
    pad = float(args.pad_deg)
    if args.clip_bbox:
        raw = args.clip_bbox.replace(",", " ").split()
        parts = [float(x) for x in raw if str(x).strip()]
        if len(parts) != 4:
            raise SystemExit(
                "--clip-bbox: need four numbers west,south,east,north (WGS84 degrees)"
            )
        wc, sc, ec, nc = parts
        west, south, east, north = wc - pad, sc - pad, ec + pad, nc + pad
    elif str(args.preset) == "solvents":
        wc, sc, ec, nc = DEFAULT_SOLVENTS_CLIP_WGS84
        west, south, east, north = wc - pad, sc - pad, ec + pad, nc + pad
    elif str(args.preset) == "waste_j" or (
        args.preset is None and _tif_suggests_waste_j(tif_path)
    ):
        wc, sc, ec, nc = DEFAULT_SOLVENTS_CLIP_WGS84
        west, south, east, north = wc - pad, sc - pad, ec + pad, nc + pad
    else:
        west, south, east, north = w - pad, s - pad, e + pad, n + pad
    gw = max(64, int(args.max_width))
    gh = max(64, int(args.max_height))
    dst_t = from_bounds(west, south, east, north, gw, gh)

    solvents_cfg_path = args.solvents_config
    if solvents_cfg_path is None:
        solvents_cfg_path = Path("Solvents/config/solvents.defaults.json")
    if not solvents_cfg_path.is_absolute():
        solvents_cfg_path = root / solvents_cfg_path

    fugitive_cfg_path = args.fugitive_config
    if fugitive_cfg_path is None:
        fugitive_cfg_path = Path("PROXY/config/fugitive/area_source.yaml")
        if str(args.preset) == "fugitive_area":
            for s in _load_sectors_config(root).get("sectors", []):
                if str(s.get("builder", "")) == "fugitive_area":
                    rel = s.get("fugitive_config")
                    if rel:
                        fugitive_cfg_path = Path(str(rel))
                    break
    if not fugitive_cfg_path.is_absolute():
        fugitive_cfg_path = root / fugitive_cfg_path

    waste_j_cfg_path = args.waste_j_config
    if waste_j_cfg_path is None:
        waste_j_cfg_path = Path("Waste/j_waste_weights/config.yaml")
    if not waste_j_cfg_path.is_absolute():
        waste_j_cfg_path = root / waste_j_cfg_path

    offroad_cfg_path = args.offroad_config
    if offroad_cfg_path is None:
        offroad_cfg_path = Path("Offroad/config/offroad_area.yaml")
        if str(args.preset) == "offroad":
            for sec in _load_sectors_config(root).get("sectors", []):
                if str(sec.get("builder", "")) == "offroad_area":
                    rel = sec.get("offroad_config")
                    if rel:
                        offroad_cfg_path = Path(str(rel))
                    break
    if not offroad_cfg_path.is_absolute():
        offroad_cfg_path = root / offroad_cfg_path

    sol_layers: list[tuple[str, np.ndarray]] = []
    sol_frag = ""
    if str(args.preset) == "solvents" or _tif_suggests_solvents(tif_path):
        import os as _os

        prev_skip = _os.environ.get("SOLVENTS_SKIP_OSM")
        if args.solvents_skip_osm:
            _os.environ["SOLVENTS_SKIP_OSM"] = "1"
        try:
            sol_layers, sol_frag = _build_solvents_context_layers(
                root,
                solvents_cfg_path,
                west,
                south,
                east,
                north,
                dst_t,
                (gh, gw),
                resampling=str(args.resampling),
            )
        finally:
            if args.solvents_skip_osm:
                if prev_skip is None:
                    _os.environ.pop("SOLVENTS_SKIP_OSM", None)
                else:
                    _os.environ["SOLVENTS_SKIP_OSM"] = prev_skip

    fug_layers: list[tuple[str, np.ndarray]] = []
    fug_frag = ""
    if str(args.preset) == "fugitive_area" or _tif_suggests_fugitive(tif_path):
        try:
            fug_layers, fug_frag = _build_fugitive_context_layers(
                root,
                fugitive_cfg_path,
                west,
                south,
                east,
                north,
                dst_t,
                (gh, gw),
                resampling=str(args.resampling),
                sectors_data=_load_sectors_config(root),
            )
        except Exception as exc:
            print(f"Warning: Fugitive context layers skipped: {exc}", file=sys.stderr)

    waste_layers: list[tuple[str, np.ndarray]] = []
    waste_frag = ""
    if str(args.preset) == "waste_j" or _tif_suggests_waste_j(tif_path):
        try:
            waste_layers, waste_frag = _build_waste_j_context_layers(
                root,
                waste_j_cfg_path,
                west,
                south,
                east,
                north,
                dst_t,
                (gh, gw),
                resampling=str(args.resampling),
                sectors_data=_load_sectors_config(root),
            )
        except Exception as exc:
            print(f"Warning: Waste J context layers skipped: {exc}", file=sys.stderr)

    ship_layers: list[tuple[str, np.ndarray]] = []
    ship_frag = ""
    if str(args.preset) == "shipping" or _tif_suggests_shipping(tif_path):
        try:
            ship_layers, ship_frag = _build_shipping_context_layers(
                root,
                west,
                south,
                east,
                north,
                dst_t,
                (gh, gw),
                resampling=str(args.resampling),
            )
        except Exception as exc:
            print(f"Warning: Shipping context layers skipped: {exc}", file=sys.stderr)

    off_layers: list[tuple[str, np.ndarray]] = []
    off_frag = ""
    if str(args.preset) == "offroad" or _tif_suggests_offroad(tif_path):
        try:
            off_layers, off_frag = _build_offroad_context_layers(
                root,
                offroad_cfg_path,
                west,
                south,
                east,
                north,
                dst_t,
                (gh, gw),
                resampling=str(args.resampling),
                sectors_data=_load_sectors_config(root),
            )
        except Exception as exc:
            print(f"Warning: Offroad context layers skipped: {exc}", file=sys.stderr)

    arr = reproject_one(
        tif_path,
        dst_t,
        (gh, gw),
        resampling=args.resampling,
        band=int(args.band),
    )

    spread_rgba: np.ndarray | None = None
    if (
        not args.no_cross_pollutant_spread
        and n_bands_file >= 2
        and (
            str(args.preset) == "residential"
            or _tif_suggests_residential_overlay(tif_path)
            or str(args.preset) == "solvents"
            or _tif_suggests_solvents(tif_path)
            or str(args.preset) == "fugitive_area"
            or _tif_suggests_fugitive(tif_path)
            or str(args.preset) == "waste_j"
            or _tif_suggests_waste_j(tif_path)
            or str(args.preset) == "offroad"
            or _tif_suggests_offroad(tif_path)
        )
    ):
        try:
            spread_rgba = _cross_pollutant_weight_spread_rgba(
                tif_path,
                dst_t,
                gh,
                gw,
                resampling=str(args.resampling),
                reproject_one=reproject_one,
            )
        except Exception as exc:
            print(f"Warning: cross-pollutant spread layer skipped: {exc}", file=sys.stderr)

    colour_mode = str(args.colour_mode)
    cmap_name = str(args.cmap)

    z01: np.ndarray | None = None
    valid_pc: np.ndarray | None = None
    cams_grid_fc: dict | None = None
    resolved_sector: str | None = None
    ag_corine_rgba: np.ndarray | None = None
    ag_wp_layers: list[tuple[str, np.ndarray]] = []
    wp_bar_bounds: tuple[float, float] | None = None
    legend_ag_codes: tuple[int, ...] | None = None
    res_corine_rgba: np.ndarray | None = None
    res_hot_layers: list[tuple[str, np.ndarray]] = []
    res_pop_rgba: np.ndarray | None = None
    res_hot_cbar_specs: list[dict] = []
    extra_res_legend = ""
    display_pollutant = _resolve_display_pollutant(sidecar_proxy, int(args.band))

    if colour_mode == "per_cams_cell":
        nc_path = args.cams_nc
        if nc_path is None:
            nc_path = _default_cams_nc(root)
        if nc_path is None or not nc_path.is_file():
            raise SystemExit(
                "per_cams_cell requires CAMS NetCDF. Pass --cams-nc or set paths.cams_nc in "
                "SourceProxies/config/sectors.json"
            )
        nc_path = nc_path if nc_path.is_absolute() else root / nc_path

        sector = str(args.cams_sector)
        if sector == "auto":
            if args.preset is None:
                raise SystemExit(
                    "per_cams_cell with --cams-sector auto needs --preset agriculture, "
                    "fugitive_area, offroad, public_power_area, public_power_point, residential, shipping, solvents, waste_j, "
                    "or pass --cams-sector explicitly"
                )
            sector = PRESET_TO_CAMS_SECTOR.get(str(args.preset))
            if sector is None:
                raise SystemExit(
                    "per_cams_cell: use a preset that maps to a CAMS area sector, or set "
                    "--cams-sector ag_kl | public_power_a | gnfr_c | gnfr_d | gnfr_e | gnfr_g | gnfr_i | gnfr_j"
                )

        rows, cols = np.indices((gh, gw))
        xs, ys = transform_xy(dst_t, rows + 0.5, cols + 0.5, offset="center")
        lons = np.asarray(xs, dtype=np.float64).reshape(gh, gw)
        lats = np.asarray(ys, dtype=np.float64).reshape(gh, gw)

        try:
            import xarray as xr
        except ImportError as exc:
            raise SystemExit("per_cams_cell needs xarray (and netCDF4). pip install xarray netCDF4") from exc

        ds = xr.open_dataset(nc_path)
        try:
            if sector == "ag_kl":
                m_area = _mask_agri_kl_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_c":
                m_area = _mask_gnfr_c_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_e":
                m_area = _mask_gnfr_e_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_d":
                m_area = _mask_gnfr_d_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_j":
                m_area = _mask_gnfr_j_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_g":
                m_area = _mask_gnfr_g_area(ds, str(args.cams_iso3).strip().upper(), root)
            elif sector == "gnfr_i":
                m_area = _mask_gnfr_i_area(ds, str(args.cams_iso3).strip().upper(), root)
            else:
                m_area = _mask_public_power_area(ds, str(args.cams_iso3).strip().upper(), root)
            cell_id = _cams_cell_id_grid(lons, lats, ds, m_area)
            resolved_sector = sector
            if not args.no_cams_grid:
                lon_src = np.asarray(ds["longitude_source"].values).ravel()
                lat_src = np.asarray(ds["latitude_source"].values).ravel()
                cams_grid_fc = _build_cams_grid_geojson(
                    ds,
                    m_area,
                    (west, south, east, north),
                    lon_src,
                    lat_src,
                )
        finally:
            ds.close()

        finite = np.isfinite(arr)
        if nodata is not None:
            finite = finite & (arr != float(nodata))
        base_valid = finite
        if args.hide_zero:
            base_valid = base_valid & (arr > 0)
        z01, valid_pc = _normalize_weights_per_cams_cell(arr, cell_id, base_valid=base_valid)

    rgba = _scalar_to_rgba(
        arr,
        colour_mode=colour_mode if colour_mode != "per_cams_cell" else "global",
        cmap_name=cmap_name,
        hide_zero=bool(args.hide_zero),
        nodata_val=float(nodata) if nodata is not None else None,
        z_precomputed_01=z01,
        valid_precomputed=valid_pc,
    )

    eff_sec = resolved_sector or _effective_cams_sector(args)
    show_ag = not args.no_ag_overlays and (
        bool(args.ag_overlays)
        or str(args.preset) == "agriculture"
        or eff_sec == "ag_kl"
        or _tif_suggests_agriculture_overlay(tif_path)
    )
    if show_ag:
        try:
            sdata = _load_sectors_config(root)
            cor_p = _resolve_corine_path(root, sdata)
            paths = sdata.get("paths") or {}
            nuts_p = Path(paths["nuts_gpkg"])
            if not nuts_p.is_absolute():
                nuts_p = root / nuts_p
            wcsv = Path(paths["weights_long"])
            if not wcsv.is_absolute():
                wcsv = root / wcsv
            cntr = str((sdata.get("country") or {}).get("nuts_cntr", "EL"))
            ag_ent = _sectors_ag_entry(sdata) or {}
            ag_codes = tuple(
                int(x) for x in (ag_ent.get("ag_clc_codes") or range(12, 23))
            )
            cor_band = int((sdata.get("corine") or {}).get("band", 1))
            if args.pollutant:
                wp_pollutants: list[str] = []
                for p in args.pollutant:
                    u = str(p).strip().upper()
                    if u:
                        wp_pollutants.append(u)
                wp_pollutants = list(dict.fromkeys(wp_pollutants))
            else:
                pols = ag_ent.get("pollutants") or ["NH3"]
                seq = pols if isinstance(pols, list) else [pols]
                wp_pollutants = [str(x).strip().upper() for x in seq if str(x).strip()]
            if not wp_pollutants:
                wp_pollutants = ["NH3"]
            if (
                cor_p is not None
                and cor_p.is_file()
                and nuts_p.is_file()
                and wcsv.is_file()
            ):
                clc = _reproject_corine_to_wgs84_grid(
                    cor_p,
                    cor_band,
                    west,
                    south,
                    east,
                    north,
                    dst_t,
                    gh,
                    gw,
                )
                clc_w = corine_grid_to_weight_codes(clc)
                ag_corine_rgba = _corine_ag_overlay_rgba(clc_w, ag_codes)
                nuts_r, nuts_ids = _rasterize_nuts2_wgs84(
                    nuts_p, cntr, dst_t, gh, gw
                )
                wps_ok: list[tuple[str, np.ndarray]] = []
                wp_viz = str(args.wp_viz)
                for pol in wp_pollutants:
                    try:
                        if wp_viz == "nuts":
                            wp = _agriculture_nuts_sum_sp_grid(
                                nuts_r, nuts_ids, wcsv, pol, ag_codes
                            )
                            if not np.any(np.isfinite(wp) & (wp > 0)):
                                print(
                                    f"Warning: NUTS2 ΣS_p grid empty for pollutant={pol!r}; "
                                    "check weights_long NUTS_ID vs GeoPackage and S_p column.",
                                    file=sys.stderr,
                                )
                        else:
                            wp = _agriculture_wp_grid(
                                clc_w, nuts_r, nuts_ids, wcsv, pol, ag_codes
                            )
                            _warn_if_wp_invisible(
                                wp,
                                pollutant=pol,
                                nuts_ids=nuts_ids,
                                weights_csv=wcsv,
                                nuts_r=nuts_r,
                                clc_w=clc_w,
                            )
                        if np.any(np.isfinite(wp) & (wp > 0)):
                            wps_ok.append((pol, wp))
                    except ValueError as exc:
                        print(
                            f"Warning: skip w_p overlay for pollutant={pol!r}: {exc}",
                            file=sys.stderr,
                        )
                pct_clip = (5.0, 95.0) if wp_viz == "nuts" else None
                bounds_wp = _wp_global_log_bounds(
                    [w for _, w in wps_ok],
                    clip_percentile=pct_clip,
                )
                if bounds_wp is not None:
                    wp_bar_bounds = bounds_wp
                for pol, wp in wps_ok:
                    rgba_wp = _wp_importance_rgba(
                        wp,
                        cmap_name=WP_IMPORTANCE_CMAP,
                        log_vmin=bounds_wp[0] if bounds_wp is not None else None,
                        log_vmax=bounds_wp[1] if bounds_wp is not None else None,
                    )
                    if np.any(rgba_wp[..., 3] > 0):
                        ag_wp_layers.append((pol, rgba_wp))
                legend_ag_codes = ag_codes
            else:
                def _ok(p: Path | None) -> bool:
                    return p is not None and p.is_file()

                print(
                    "Warning: agriculture overlays not added (need CORINE GeoTIFF, NUTS gpkg, "
                    "and weights_long.csv). Status:",
                    file=sys.stderr,
                )
                print(
                    f"  corine: {_ok(cor_p)} {cor_p if cor_p is not None else '(none — see paths below)'}",
                    file=sys.stderr,
                )
                cfg_cor = paths.get("corine")
                print("  CORINE search order (place the U2018 CLC2018 100m GeoTIFF at one of these):", file=sys.stderr)
                for cand in iter_corine_search_paths(root, cfg_cor):
                    mark = "exists" if cand.is_file() else "missing"
                    print(f"    [{mark}] {cand}", file=sys.stderr)
                print(
                    f"  nuts_gpkg: {_ok(nuts_p)} {nuts_p}",
                    file=sys.stderr,
                )
                print(
                    f"  weights_long: {_ok(wcsv)} {wcsv}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"Warning: agriculture overlays skipped: {exc}", file=sys.stderr)

    spread_leg_extra = ""
    if spread_rgba is not None and n_bands_file >= 2:
        sg = _css_linear_gradient_stops(_sample_cmap_hex("cividis", n=11))
        spread_leg_extra = f"""
<details class="pl-details">
  <summary>Cross-pollutant weight spread</summary>
  <p class="pl-hint">Per-pixel <b>standard deviation</b> across <b>{n_bands_file}</b> weight bands.
  Brighter = relative weights differ more between pollutants (2-98% percentile stretch, cividis).</p>
  <div class="pl-grad" style="background:{sg};"></div>
  <div class="pl-grad-lbl"><span>low</span><span>high</span></div>
</details>"""

    show_residential = not args.no_residential_overlays and (
        str(args.preset) == "residential"
        or bool(args.residential_overlays)
        or _tif_suggests_residential_overlay(tif_path)
    )
    if show_residential:
        try:
            rc_path = args.residential_config
            if rc_path is None:
                rc_path = DEFAULT_RESIDENTIAL_CONFIG
            rc_path = rc_path if rc_path.is_absolute() else root / rc_path
            res_cfg = _load_json_if_exists(rc_path)
            sdata = _load_sectors_config(root)
            (
                res_corine_rgba,
                res_hot_layers,
                res_pop_rgba,
                res_hot_cbar_specs,
            ) = _build_residential_context_layers(
                root,
                west=west,
                south=south,
                east=east,
                north=north,
                dst_t=dst_t,
                gh=gh,
                gw=gw,
                res_cfg=res_cfg,
                sectors_data=sdata,
                resampling=str(args.resampling),
            )
            extra_res_legend = _residential_legend_html(
                show_corine_l1=True,
                hotmap_titles=[t for t, _ in res_hot_layers],
                pop_ok=res_pop_rgba is not None,
                hot_cbar_specs=res_hot_cbar_specs,
                spread_legend_fragment=spread_leg_extra,
            )
        except Exception as exc:
            print(f"Warning: residential overlays skipped: {exc}", file=sys.stderr)
    elif spread_leg_extra:
        extra_res_legend = spread_leg_extra

    out = args.out_html
    if out is None:
        out = tif_path.with_name(tif_path.stem + "_proxy_map.html")
    elif not out.is_absolute():
        out = root / out

    fmap = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=8,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        "CartoDB positron",
        name="Light (CartoDB Positron)",
        control=True,
    ).add_to(fmap)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles &copy; Esri &mdash; "
            "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
        ),
        name="Satellite (Esri World Imagery)",
        max_zoom=19,
        control=True,
    ).add_to(fmap)

    if display_pollutant:
        layer_name = (
            f"Residential weights · {display_pollutant} (band {int(args.band)}/{n_bands_file})"
        )
    elif str(args.preset) == "solvents" or _tif_suggests_solvents(tif_path):
        layer_name = (
            f"Solvents GNFR E area · {tif_path.name} (band {int(args.band)}/{n_bands_file})"
        )
    elif str(args.preset) == "fugitive_area" or _tif_suggests_fugitive(tif_path):
        layer_name = (
            f"Fugitive GNFR D area · {tif_path.name} (band {int(args.band)}/{n_bands_file})"
        )
    elif str(args.preset) == "waste_j" or _tif_suggests_waste_j(tif_path):
        layer_name = (
            f"Waste GNFR J area · {tif_path.name} (band {int(args.band)}/{n_bands_file})"
        )
    elif str(args.preset) == "offroad" or _tif_suggests_offroad(tif_path):
        layer_name = (
            f"Offroad GNFR I area · {tif_path.name} (band {int(args.band)}/{n_bands_file})"
        )
    else:
        layer_name = f"Weights ({tif_path.name} b{args.band})"
    fg = folium.FeatureGroup(name=layer_name, show=True)
    weight_opacity = float(args.opacity)
    if ag_corine_rgba is not None or res_corine_rgba is not None:
        weight_opacity = min(weight_opacity, 0.78)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        mercator_project=True,
        opacity=weight_opacity,
        name="Proxy raster",
        interactive=False,
        cross_origin=False,
    ).add_to(fg)
    fg.add_to(fmap)

    if cams_grid_fc is not None and cams_grid_fc.get("features"):
        cams_grid_name = "CAMS area grid (cell outlines)"
        if resolved_sector == "gnfr_c":
            cams_grid_name = "CAMS GNFR C area grid (cell outlines)"
        elif resolved_sector == "gnfr_e":
            cams_grid_name = "CAMS GNFR E area grid (cell outlines)"
        elif resolved_sector == "gnfr_d":
            cams_grid_name = "CAMS GNFR D area grid (cell outlines)"
        elif resolved_sector == "gnfr_j":
            cams_grid_name = "CAMS GNFR J area grid (cell outlines)"
        elif resolved_sector == "gnfr_g":
            cams_grid_name = "CAMS GNFR G area grid (cell outlines)"
        elif resolved_sector == "gnfr_i":
            cams_grid_name = "CAMS GNFR I area grid (cell outlines)"
        elif resolved_sector == "ag_kl":
            cams_grid_name = "CAMS ag/forest area grid (cell outlines)"
        fg_cams = folium.FeatureGroup(
            name=cams_grid_name,
            show=True,
        )
        folium.GeoJson(
            cams_grid_fc,
            style_function=lambda _f: {
                "fillColor": "#00000000",
                "color": "#0d47a1",
                "weight": 1.25,
                "fillOpacity": 0.0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["cams_source_index", "lon_c", "lat_c"],
                aliases=["CAMS source index", "Centre lon", "Centre lat"],
                sticky=True,
            ),
        ).add_to(fg_cams)
        fg_cams.add_to(fmap)

    if ag_corine_rgba is not None:
        fg_cor = folium.FeatureGroup(
            name="CORINE agricultural classes (CLC 12–22, vivid)",
            show=False,
        )
        folium.raster_layers.ImageOverlay(
            image=ag_corine_rgba,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=1.0,
            name="CORINE ag",
            interactive=False,
            cross_origin=False,
        ).add_to(fg_cor)
        fg_cor.add_to(fmap)

    wp_viz_lbl = "NUTS2 ΣS_p" if str(args.wp_viz) == "nuts" else "NUTS×CLC w_p"
    for pol, rgba_wp in ag_wp_layers:
        fg_wp = folium.FeatureGroup(
            name=f"w_p · {pol} ({wp_viz_lbl}, YlOrRd)",
            show=False,
        )
        folium.raster_layers.ImageOverlay(
            image=rgba_wp,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.78,
            name=f"w_p {pol}",
            interactive=False,
            cross_origin=False,
        ).add_to(fg_wp)
        fg_wp.add_to(fmap)

    if res_corine_rgba is not None:
        fg_rc = folium.FeatureGroup(
            name="CORINE L1 · codes 1-3 (urban / industry)",
            show=True,
        )
        folium.raster_layers.ImageOverlay(
            image=res_corine_rgba,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.72,
            name="CORINE urban",
            interactive=False,
            cross_origin=False,
        ).add_to(fg_rc)
        fg_rc.add_to(fmap)

    for title, rgba_hm in res_hot_layers:
        cmap_lbl = (
            "viridis"
            if "heat" in title.lower()
            else ("plasma" if "gfa" in title.lower() else "viridis")
        )
        fg_hm = folium.FeatureGroup(
            name=f"{title} (log, {cmap_lbl})",
            show=False,
        )
        folium.raster_layers.ImageOverlay(
            image=rgba_hm,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_hm)
        fg_hm.add_to(fmap)

    if res_pop_rgba is not None:
        fg_pop = folium.FeatureGroup(
            name="LandScan population (log, YlOrRd)",
            show=False,
        )
        folium.raster_layers.ImageOverlay(
            image=res_pop_rgba,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.78,
            name="LandScan",
            interactive=False,
            cross_origin=False,
        ).add_to(fg_pop)
        fg_pop.add_to(fmap)

    if spread_rgba is not None:
        fg_sp = folium.FeatureGroup(
            name=f"Cross-pollutant weight spread (stdev, {n_bands_file} bands)",
            show=False,
        )
        folium.raster_layers.ImageOverlay(
            image=spread_rgba,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.72,
            name="stdev across pollutants",
            interactive=False,
            cross_origin=False,
        ).add_to(fg_sp)
        fg_sp.add_to(fmap)

    for title, rgba_sl in sol_layers:
        fg_sl = folium.FeatureGroup(name=f"[proxy input] {title}", show=False)
        folium.raster_layers.ImageOverlay(
            image=rgba_sl,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_sl)
        fg_sl.add_to(fmap)

    for title, rgba_fl in fug_layers:
        fg_fl = folium.FeatureGroup(name=f"[proxy input] {title}", show=False)
        folium.raster_layers.ImageOverlay(
            image=rgba_fl,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_fl)
        fg_fl.add_to(fmap)

    for title, rgba_wj in waste_layers:
        fg_wj = folium.FeatureGroup(name=f"[proxy input] {title}", show=False)
        folium.raster_layers.ImageOverlay(
            image=rgba_wj,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_wj)
        fg_wj.add_to(fmap)

    for title, rgba_sh in ship_layers:
        fg_sh = folium.FeatureGroup(name=f"[proxy input] {title}", show=False)
        folium.raster_layers.ImageOverlay(
            image=rgba_sh,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_sh)
        fg_sh.add_to(fmap)

    for title, rgba_of in off_layers:
        fg_of = folium.FeatureGroup(name=f"[proxy input] {title}", show=False)
        folium.raster_layers.ImageOverlay(
            image=rgba_of,
            bounds=[[south, west], [north, east]],
            mercator_project=True,
            opacity=0.82,
            name=title,
            interactive=False,
            cross_origin=False,
        ).add_to(fg_of)
        fg_of.add_to(fmap)

    mode_caption_long = {
        "global": "Linear stretch (global min–max of valid pixels)",
        "log": "log10 then linear stretch",
        "percentile": "2–98% percentile stretch",
        "per_cams_cell": "Intensity 0–1 within each CAMS area cell (relative share contrast)",
    }.get(colour_mode, colour_mode)
    mode_caption_short = {
        "global": "Global min–max stretch",
        "log": "log10 + linear stretch",
        "percentile": "2–98% percentiles",
        "per_cams_cell": "Per CAMS cell 0–1",
    }.get(colour_mode, colour_mode)

    colours = _sample_cmap_hex(cmap_name, n=11)
    wp_names_leg = [p for p, _ in ag_wp_layers] if ag_wp_layers else None
    active_band_html = ""
    if display_pollutant:
        active_band_html = (
            f'<div class="pl-active-band"><p class="pl-meta" style="font-weight:650;margin:0 0 6px 0;">'
            f"Displayed pollutant: <b>{html.escape(display_pollutant)}</b> "
            f"&middot; band <b>{int(args.band)}</b> / <b>{n_bands_file}</b></p></div>"
        )
    elif str(args.preset) == "solvents" or _tif_suggests_solvents(tif_path):
        try:
            scfg = json.loads(solvents_cfg_path.read_text(encoding="utf-8"))
            pols = scfg.get("pollutants") or []
            bi = int(args.band) - 1
            if 0 <= bi < len(pols):
                active_band_html = (
                    f'<div class="pl-active-band"><p class="pl-meta" style="font-weight:650;margin:0 0 6px 0;">'
                    f"Solvents pollutant band: <b>{html.escape(str(pols[bi]))}</b> "
                    f"&middot; <b>{int(args.band)}</b> / <b>{n_bands_file}</b></p></div>"
                )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    elif str(args.preset) == "fugitive_area" or _tif_suggests_fugitive(tif_path):
        try:
            from PROXY.core.dataloaders import load_yaml

            fyaml = load_yaml(fugitive_cfg_path)
            pols = fyaml.get("pollutants") or []
            bi = int(args.band) - 1
            if 0 <= bi < len(pols):
                active_band_html = (
                    f'<div class="pl-active-band"><p class="pl-meta" style="font-weight:650;margin:0 0 6px 0;">'
                    f"Fugitive pollutant band: <b>{html.escape(str(pols[bi]))}</b> "
                    f"&middot; <b>{int(args.band)}</b> / <b>{n_bands_file}</b></p></div>"
                )
        except Exception:
            pass
    elif str(args.preset) == "waste_j" or _tif_suggests_waste_j(tif_path):
        try:
            wyaml = yaml.safe_load(waste_j_cfg_path.read_text(encoding="utf-8")) or {}
            pols = wyaml.get("cams", {}).get("pollutants_nc") or []
            bi = int(args.band) - 1
            if 0 <= bi < len(pols):
                active_band_html = (
                    f'<div class="pl-active-band"><p class="pl-meta" style="font-weight:650;margin:0 0 6px 0;">'
                    f"Waste J pollutant band: <b>{html.escape(str(pols[bi]))}</b> "
                    f"&middot; <b>{int(args.band)}</b> / <b>{n_bands_file}</b></p></div>"
                )
        except Exception:
            pass
    elif str(args.preset) == "offroad" or _tif_suggests_offroad(tif_path):
        try:
            oyaml = yaml.safe_load(offroad_cfg_path.read_text(encoding="utf-8")) or {}
            pols = oyaml.get("pollutants") or []
            bi = int(args.band) - 1
            if 0 <= bi < len(pols):
                active_band_html = (
                    f'<div class="pl-active-band"><p class="pl-meta" style="font-weight:650;margin:0 0 6px 0;">'
                    f"Offroad pollutant band: <b>{html.escape(str(pols[bi]))}</b> "
                    f"&middot; <b>{int(args.band)}</b> / <b>{n_bands_file}</b></p></div>"
                )
        except Exception:
            pass

    sol_map_leg = ""
    if sol_frag.strip() or sol_layers:
        sol_map_leg = (sol_frag or "") + _solvents_legend_block()

    fug_map_leg = ""
    if fug_frag.strip() or fug_layers:
        fug_map_leg = (fug_frag or "") + _fugitive_legend_block()

    waste_map_leg = ""
    if waste_frag.strip() or waste_layers:
        waste_map_leg = (waste_frag or "") + _waste_j_legend_block()

    ship_map_leg = ""
    if ship_frag.strip() or ship_layers:
        ship_map_leg = (ship_frag or "") + _shipping_legend_block()

    offroad_map_leg = ""
    if off_frag.strip() or off_layers:
        offroad_map_leg = (off_frag or "") + _offroad_legend_block()

    fmap.get_root().html.add_child(
        folium.Element(
            _combined_map_legend_html(
                tif_name=tif_path.name,
                band=int(args.band),
                colour_mode=colour_mode,
                cmap_name=cmap_name,
                mode_caption_long=mode_caption_long,
                mode_caption_short=mode_caption_short,
                gradient_hex=colours,
                ag_codes=legend_ag_codes,
                wp_pollutant_names=wp_names_leg,
                wp_log_range=wp_bar_bounds,
                wp_cmap_legend=WP_IMPORTANCE_CMAP,
                wp_viz=str(args.wp_viz),
                extra_legend_html=(
                    extra_res_legend
                    + sol_map_leg
                    + fug_map_leg
                    + waste_map_leg
                    + ship_map_leg
                    + offroad_map_leg
                ),
                active_band_html=active_band_html,
            )
        )
    )

    if wp_bar_bounds is not None and ag_wp_layers:
        wp_bar_hex = _sample_cmap_hex(WP_IMPORTANCE_CMAP, n=11)
        if str(args.wp_viz) == "nuts":
            cbar_cap = "log10( sum S_p per NUTS2 )  YlOrRd  (ag CLCs; 5–95% clip; shared)"
        else:
            cbar_cap = "log10( w_p )  YlOrRd  (shared for all pollutants)"
        br_wp = LinearColormap(
            wp_bar_hex,
            vmin=wp_bar_bounds[0],
            vmax=wp_bar_bounds[1],
            caption=cbar_cap,
        )
        br_wp.add_to(fmap)

    try:
        from folium.plugins import Fullscreen

        Fullscreen(position="bottomright", title="Fullscreen", title_cancel="Exit").add_to(fmap)
    except Exception:
        pass

    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
