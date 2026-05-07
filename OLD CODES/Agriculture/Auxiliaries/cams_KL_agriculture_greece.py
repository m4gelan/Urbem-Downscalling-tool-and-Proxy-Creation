#!/usr/bin/env python3
"""
Explore CAMS-REG-ANT v8.1 emissions for GNFR K (AgriLivestock) and L (AgriOther):
area vs point sources, Greece (or full domain). Mirrors the workflow of
``PublicPower/Auxiliaries/cams_A_publicpower_greece.py`` but without JRC overlays.

``emission_category_index`` is 1-based over 15 GNFR codes:
  A, B, C, D, E, F1, F2, F3, F4, G, H, I, J, K, L
so K_AgriLivestock = 14 and L_AgriOther = 15 (see ``urbem_interface.emissions.prepare_cams``).

Usage (from project root):
  python Agriculture/Auxiliaries/cams_KL_agriculture_greece.py
  python Agriculture/Auxiliaries/cams_KL_agriculture_greece.py --sector K
  python Agriculture/Auxiliaries/cams_KL_agriculture_greece.py --all-countries --no-map
  python Agriculture/Auxiliaries/cams_KL_agriculture_greece.py --pollutant nh3 --out-map Agriculture/outputs/cams_KL_GRC_nh3.html
  python Agriculture/Auxiliaries/cams_KL_agriculture_greece.py --map-scale linear

Requires: xarray, netCDF4, pandas; for maps: folium, branca
"""

from __future__ import annotations

import argparse
from typing import Any
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 1-based indices in CAMS-REG-ANT v8.1 NetCDF
IDX_K_AGRI_LIVESTOCK = 14
IDX_L_AGRI_OTHER = 15

DEFAULT_NC = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "given_CAMS"
    / "CAMS-REG-ANT_v8.1_TNO_ftp"
    / "netcdf"
    / "CAMS-REG-v8_1_emissions_year2019.nc"
)

SKIP_VARS = frozenset(
    {
        "longitude_source",
        "latitude_source",
        "longitude_index",
        "latitude_index",
        "country_index",
        "emission_category_index",
        "source_type_index",
    }
)

SECTOR_LABEL = {
    IDX_K_AGRI_LIVESTOCK: "K_AgriLivestock",
    IDX_L_AGRI_OTHER: "L_AgriOther",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pollutant_vars(ds) -> list[str]:
    out: list[str] = []
    for name, v in ds.data_vars.items():
        if name in SKIP_VARS:
            continue
        dims = tuple(v.dims)
        if len(dims) != 1 or dims[0] != "source":
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        out.append(name)
    return sorted(out)


def _decode_country_ids(ds) -> list[str]:
    raw = ds["country_id"].values
    out: list[str] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "replace").strip())
        else:
            out.append(str(x).strip())
    return out


def _country_index_1based(ds, iso3: str) -> int:
    codes = _decode_country_ids(ds)
    u = iso3.strip().upper()
    try:
        return codes.index(u) + 1
    except ValueError as exc:
        raise SystemExit(f"Country {iso3!r} not in NetCDF country_id (have {len(codes)} countries).") from exc


def _build_domain_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    country_idx: np.ndarray,
    country_1based: int | None,
    bbox: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if country_1based is None:
        m = np.ones(country_idx.shape[0], dtype=bool)
    else:
        m = country_idx == country_1based
    if bbox is not None:
        lon0, lat0, lon1, lat1 = bbox
        m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    return m


def _summarize_pollutants(
    ds,
    base_mask: np.ndarray,
    pollutants: list[str],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, str]]:
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    area_m = base_mask & (st == 1)
    point_m = base_mask & (st == 2)
    units: dict[str, str] = {}
    rows: dict[str, tuple[float, float, float]] = {}
    for name in pollutants:
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        if arr.size != base_mask.size:
            continue
        ta = float(np.nansum(arr[area_m]))
        tp = float(np.nansum(arr[point_m]))
        rows[name] = (ta, tp, ta + tp)
        u = ds[name].attrs.get("units", "kg/year")
        lu = ds[name].attrs.get("long_units", "")
        units[name] = f"{u}" + (f" ({lu})" if lu else "")
    return rows, units


def _point_positive_counts(
    ds,
    point_mask: np.ndarray,
    pollutants: list[str],
) -> tuple[dict[str, int], dict[str, float]]:
    """Per pollutant: count of point rows with value > 0, and sum on those rows."""
    n_pos: dict[str, int] = {}
    sums: dict[str, float] = {}
    for name in pollutants:
        arr = np.asarray(ds[name].values).ravel().astype(np.float64)
        if arr.size != point_mask.size:
            continue
        m = point_mask & np.isfinite(arr) & (arr > 0)
        n_pos[name] = int(np.count_nonzero(m))
        sums[name] = float(np.nansum(arr[m]))
    return n_pos, sums


def _fmt(x: float) -> str:
    if not math.isfinite(x) or x == 0.0:
        return f"{x:.6g}"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.4f}e9"
    if ax >= 1e6:
        return f"{x/1e6:.4f}e6"
    if ax >= 1e3:
        return f"{x/1e3:.4f}e3"
    return f"{x:.6g}"


def print_comparison_table(
    totals: dict[str, tuple[float, float, float]],
    units: dict[str, str],
    *,
    title: str,
) -> None:
    print(title)
    print(
        f"{'pollutant':<12} {'area':>14} {'point':>14} {'total':>14}  "
        f"point_share%  units"
    )
    print("-" * 88)
    for name in sorted(totals.keys()):
        ta, tp, tt = totals[name]
        share = (100.0 * tp / tt) if tt > 0 else float("nan")
        u = units.get(name, "")
        print(
            f"{name:<12} {_fmt(ta):>14} {_fmt(tp):>14} {_fmt(tt):>14}  "
            f"{share:10.2f}  {u}"
        )


def print_point_pollutant_matrix(
    sectors_data: dict[int, tuple[dict[str, int], dict[str, float]]],
    pollutants: list[str],
    *,
    title: str,
) -> None:
    """For each sector: which pollutants have point sources with emissions > 0."""
    print()
    print(title)
    for sidx in sorted(sectors_data.keys()):
        n_pos, sums = sectors_data[sidx]
        label = SECTOR_LABEL.get(sidx, str(sidx))
        print(f"  {label} (emission_category_index={sidx})")
        print(f"    {'pollutant':<12} {'n_point>0':>12} {'point_sum_kg_yr':>18}")
        for pol in sorted(pollutants):
            if pol not in n_pos:
                continue
            n = n_pos[pol]
            t = sums.get(pol, 0.0)
            flag = "" if n == 0 else "  <-- point emissions"
            print(f"    {pol:<12} {n:>12} {_fmt(t):>18}{flag}")


def add_cams_sector_layers_to_map(
    fmap,
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    *,
    sector_index: int,
    sector_short: str,
    area_group_name: str,
    point_group_name: str,
    outline_area: str,
    outline_point: str,
    map_max_area_cells: int,
    map_sample_seed: int,
    vmin_val: float,
    vmax_val: float,
    map_scale: str,
) -> Any:
    try:
        import folium
        from branca.colormap import LinearColormap
    except ImportError as exc:
        raise SystemExit("folium (and branca) are required for maps. pip install folium") from exc

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    sector_m = emis == sector_index
    m_area = base_mask & sector_m & (st == 1)
    m_point = base_mask & sector_m & (st == 2)

    want_log = map_scale.strip().lower() == "log"
    use_log = want_log and vmin_val > 0 and vmax_val > 0 and math.isfinite(vmin_val) and math.isfinite(vmax_val)
    if use_log:
        c_vmin = math.log10(vmin_val)
        c_vmax = math.log10(vmax_val)
        if c_vmin >= c_vmax:
            c_vmax = c_vmin + 1e-9
        cap = f"{pollutant} — log10(kg/year)"
    else:
        c_vmin = float(vmin_val)
        c_vmax = float(vmax_val)
        if c_vmin >= c_vmax:
            c_vmax = c_vmin + 1e-9 if c_vmin > 0 else 1.0
        if want_log and not use_log:
            cap = f"{pollutant} (kg/year) — linear scale (no positive range for log)"
        else:
            cap = f"{pollutant} (kg/year)"

    colors = ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"]
    cmap = LinearColormap(colors, vmin=c_vmin, vmax=c_vmax, caption=cap)

    def norm_radius_t(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 0.0
        if use_log:
            lv = math.log10(max(v, vmin_val))
            lv = max(c_vmin, min(lv, c_vmax))
            return (lv - c_vmin) / max(c_vmax - c_vmin, 1e-15)
        return max(0.0, min(1.0, v / max(vmax_val, 1e-30)))

    def radius_for(v: float) -> float:
        t = norm_radius_t(v)
        if t <= 0:
            return 3.0
        return min(28.0, 4.0 + 18.0 * math.sqrt(t))

    def color_for(v: float) -> str:
        if not math.isfinite(v) or v <= 0:
            return "#cccccc"
        if use_log:
            x = math.log10(max(v, vmin_val))
            x = max(c_vmin, min(x, c_vmax))
        else:
            x = max(c_vmin, min(v, c_vmax))
        return cmap(x)

    fg_area = folium.FeatureGroup(name=area_group_name, show=True)
    fg_point = folium.FeatureGroup(name=point_group_name, show=True)

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

    idx_area = np.flatnonzero(m_area)
    n_area_all = int(idx_area.size)
    if map_max_area_cells > 0 and n_area_all > map_max_area_cells:
        rng = np.random.default_rng(map_sample_seed)
        idx_area = np.sort(rng.choice(idx_area, size=map_max_area_cells, replace=False))

    for i in idx_area:
        v = float(vals[i])
        if not math.isfinite(v):
            continue
        color = color_for(v)
        li, ji = int(lon_ii[i]), int(lat_ii[i])
        west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
        south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
        if south > north:
            south, north = north, south
        if west > east:
            west, east = east, west
        pop = (
            f"<b>{html.escape(sector_short)}</b> area<br/>"
            f"<b>{html.escape(pollutant)}</b> {v:.6g} kg/yr<br/>"
            f"cell lon [{west:.4f}, {east:.4f}]<br/>"
            f"cell lat [{south:.4f}, {north:.4f}]<br/>"
            f"centre {lon[i]:.4f}, {lat[i]:.4f}"
        )
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=outline_area,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.45,
            popup=folium.Popup(pop, max_width=280),
        ).add_to(fg_area)

    for i in np.flatnonzero(m_point):
        v = float(vals[i])
        if not math.isfinite(v):
            continue
        color = color_for(v)
        r = radius_for(v)
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=r,
            color=outline_point,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{html.escape(sector_short)}</b> point<br/>"
                f"<b>{html.escape(pollutant)}</b> {v:.6g} kg/yr<br/>"
                f"lon {lon[i]:.4f}, lat {lat[i]:.4f}",
                max_width=260,
            ),
        ).add_to(fg_point)

    fg_area.add_to(fmap)
    fg_point.add_to(fmap)
    return cmap


def _map_value_range(
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    sector_indices: list[int],
) -> tuple[float, float]:
    lon = np.asarray(ds["longitude_source"].values).ravel()
    _ = lon  # same length as all per-source arrays
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    sec_m = np.isin(emis, np.array(sector_indices, dtype=np.int64))
    m = base_mask & sec_m & (st <= 2)
    pos = vals > 0
    vpos = vals[pos & m]
    if vpos.size:
        vmin = float(np.min(vpos))
        vmax = float(np.max(vpos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
        return vmin, vmax
    return 0.0, 1.0


def build_folium_map(
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    sector_indices: list[int],
    *,
    out_html: Path,
    map_center_latlon: tuple[float, float] | None,
    zoom_start: int,
    map_max_area_cells: int,
    map_sample_seed: int,
    map_scale: str,
) -> None:
    try:
        import folium
    except ImportError as exc:
        raise SystemExit("folium (and branca) are required for --out-map. pip install folium") from exc

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    if map_center_latlon is not None:
        center_lat, center_lon = map_center_latlon
    else:
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        sec_m = np.isin(emis, np.array(sector_indices, dtype=np.int64))
        use = base_mask & sec_m
        if not np.any(use):
            center_lat, center_lon = 39.0, 22.0
        else:
            center_lat = float(np.mean(lat[use]))
            center_lon = float(np.mean(lon[use]))

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=int(zoom_start), tiles="CartoDB positron")
    vmin, vmax = _map_value_range(ds, base_mask, pollutant, sector_indices)
    ms = map_scale.strip().lower()
    if ms == "log" and vmin > 0 and vmax > 0:
        print(
            f"Map color scale: log10(kg/year) over emission range "
            f"[{_fmt(vmin)}, {_fmt(vmax)}] kg/year"
        )
    elif ms == "log":
        print("Map color scale: linear fallback (no positive emissions in domain for log scale).")
    else:
        print(f"Map color scale: linear kg/year over [{_fmt(vmin)}, {_fmt(vmax)}]")

    cmap_ref = None
    style = {
        IDX_K_AGRI_LIVESTOCK: {
            "short": "K AgriLivestock",
            "area_outline": "#1f4e79",
            "point_outline": "#2e75b6",
        },
        IDX_L_AGRI_OTHER: {
            "short": "L AgriOther",
            "area_outline": "#7030a0",
            "point_outline": "#9f4fd6",
        },
    }
    for sidx in sector_indices:
        st = style.get(
            sidx,
            {"short": f"sector {sidx}", "area_outline": "#333333", "point_outline": "#111111"},
        )
        label = SECTOR_LABEL.get(sidx, str(sidx))
        cmap = add_cams_sector_layers_to_map(
            fmap,
            ds,
            base_mask,
            pollutant,
            sector_index=sidx,
            sector_short=st["short"],
            area_group_name=f"{label} - area (grid cell)",
            point_group_name=f"{label} - point sources",
            outline_area=st["area_outline"],
            outline_point=st["point_outline"],
            map_max_area_cells=map_max_area_cells,
            map_sample_seed=map_sample_seed,
            vmin_val=vmin,
            vmax_val=vmax,
            map_scale=map_scale,
        )
        cmap_ref = cmap

    if cmap_ref is not None:
        cmap_ref.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    root = _project_root()
    p = argparse.ArgumentParser(
        description="CAMS-REG v8.1: K_AgriLivestock & L_AgriOther - area vs point and point-by-pollutant summary."
    )
    p.add_argument("--nc", type=Path, default=DEFAULT_NC, help="Path to CAMS-REG emissions NetCDF")
    p.add_argument(
        "--all-countries",
        action="store_true",
        help="All countries in file; optional --bbox still applies",
    )
    p.add_argument("--country", default="GRC", help="ISO3 when not using --all-countries (default GRC)")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional WGS84 bounding box",
    )
    p.add_argument(
        "--sector",
        choices=("K", "L", "KL"),
        default="KL",
        help="K=AgriLivestock only, L=AgriOther only, KL=both (default KL)",
    )
    p.add_argument(
        "--pollutant",
        default="nh3",
        help="Pollutant variable for the map (default nh3)",
    )
    p.add_argument("--no-map", action="store_true", help="Skip Folium HTML")
    p.add_argument(
        "--out-map",
        type=Path,
        default=None,
        help="Folium HTML (default: Agriculture/outputs/cams_KL_<ISO|ALL>_<pollutant>.html)",
    )
    p.add_argument(
        "--map-max-area-cells",
        type=int,
        default=25_000,
        help="Max area rectangles per sector on the map (0 = all)",
    )
    p.add_argument("--map-sample-seed", type=int, default=42, help="RNG seed for area subsampling")
    p.add_argument(
        "--map-scale",
        choices=("log", "linear"),
        default="log",
        help="Color scale for map fills: log10(kg/year) (default) or linear kg/year",
    )
    p.add_argument("--map-zoom", type=int, default=None, help="Initial zoom (default 4 if all countries else 7)")
    p.add_argument("--map-center-lat", type=float, default=None)
    p.add_argument("--map-center-lon", type=float, default=None)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="CSV: area/point/total per pollutant for each selected sector (long format)",
    )
    p.add_argument(
        "--out-points-csv",
        type=Path,
        default=None,
        help="Optional CSV of point sources (lon, lat, sector, pollutant columns)",
    )
    args = p.parse_args()
    nc_path = args.nc if args.nc.is_absolute() else root / args.nc
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    if args.sector == "K":
        sector_indices = [IDX_K_AGRI_LIVESTOCK]
    elif args.sector == "L":
        sector_indices = [IDX_L_AGRI_OTHER]
    else:
        sector_indices = [IDX_K_AGRI_LIVESTOCK, IDX_L_AGRI_OTHER]

    import xarray as xr

    ds = xr.open_dataset(nc_path)
    try:
        pollutants = _pollutant_vars(ds)
        if not pollutants:
            raise SystemExit("No pollutant variables with dimension 'source' found.")

        country_1b: int | None
        if args.all_countries:
            country_1b = None
        else:
            country_1b = _country_index_1based(ds, args.country)

        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)

        bbox_t = tuple(args.bbox) if args.bbox else None
        sector_m = np.isin(emis, np.array(sector_indices, dtype=np.int64))
        domain = _build_domain_mask(lon, lat, ci, country_1b, bbox_t)
        base = sector_m & domain
        n = int(np.count_nonzero(base))
        if n == 0:
            raise SystemExit("No K/L agricultural sources match country/bbox filter.")

        hist = str(ds.attrs.get("history", nc_path.name))
        print(f"NetCDF: {nc_path.name}")
        print(f"History: {hist[:120]}{'...' if len(hist) > 120 else ''}")
        sectors_str = ", ".join(f"{SECTOR_LABEL[s]} (index {s})" for s in sector_indices)
        if args.all_countries:
            print(f"Filter: {sectors_str}; all countries")
        else:
            print(f"Filter: {sectors_str}; country {args.country.upper()} (country_index={country_1b})")
        if args.bbox:
            print(f"  + bbox lon[{args.bbox[0]},{args.bbox[2]}] lat[{args.bbox[1]},{args.bbox[3]}]")
        print(f"Sources in domain: {n} (area {int(np.count_nonzero(base & (st == 1)))}, point {int(np.count_nonzero(base & (st == 2)))})")
        print()

        for sidx in sector_indices:
            sm = (emis == sidx) & domain
            print(
                f"  {SECTOR_LABEL[sidx]}: sources {int(np.count_nonzero(sm))} "
                f"(area {int(np.count_nonzero(sm & (st == 1)))}, "
                f"point {int(np.count_nonzero(sm & (st == 2)))})"
            )
        print()

        sectors_point_matrix: dict[int, tuple[dict[str, int], dict[str, float]]] = {}
        for sidx in sector_indices:
            sm = (emis == sidx) & domain
            pm = sm & (st == 2)
            sectors_point_matrix[sidx] = _point_positive_counts(ds, pm, pollutants)

        print_point_pollutant_matrix(
            sectors_point_matrix,
            pollutants,
            title="Point sources: count with emission > 0 per pollutant (and point total kg/yr)",
        )

        all_units: dict[str, str] = {}
        for sidx in sector_indices:
            sm = (emis == sidx) & domain
            totals, units = _summarize_pollutants(ds, sm, pollutants)
            all_units.update(units)
            print_comparison_table(
                totals,
                units,
                title=f"Totals by pollutant — {SECTOR_LABEL[sidx]} (area vs point)",
            )
            print()

        pol = args.pollutant.strip().lower()
        if pol not in pollutants:
            raise SystemExit(f"Unknown pollutant {pol!r}. Examples: {', '.join(pollutants[:8])}")

        if not args.no_map:
            out_map = args.out_map
            out_tag = "ALL" if args.all_countries else args.country.upper()
            if out_map is None:
                out_map = root / "Agriculture" / "outputs" / f"cams_KL_{out_tag}_{pol}.html"
            else:
                out_map = out_map if out_map.is_absolute() else root / out_map

            mclat, mclon = args.map_center_lat, args.map_center_lon
            if (mclat is None) ^ (mclon is None):
                raise SystemExit("Set both --map-center-lat and --map-center-lon, or neither.")
            if mclat is not None:
                map_center = (float(mclat), float(mclon))
            elif args.all_countries:
                map_center = (51.0, 14.0)
            else:
                map_center = None

            zoom_use = args.map_zoom if args.map_zoom is not None else (4 if args.all_countries else 7)
            max_area = max(0, int(args.map_max_area_cells))

            n_area_total = 0
            for sidx in sector_indices:
                sm = (emis == sidx) & domain
                n_area_total += int(np.count_nonzero(sm & (st == 1)))
            if max_area > 0 and n_area_total > max_area * len(sector_indices):
                print(
                    f"Map: up to {max_area} area cells drawn per sector "
                    f"(--map-max-area-cells); totals use all cells."
                )

            build_folium_map(
                ds,
                domain & np.isin(emis, np.array(sector_indices, dtype=np.int64)),
                pol,
                sector_indices,
                out_html=out_map,
                map_center_latlon=map_center,
                zoom_start=int(zoom_use),
                map_max_area_cells=max_area,
                map_sample_seed=int(args.map_sample_seed),
                map_scale=str(args.map_scale),
            )
            print(f"Interactive map ({pol}): {out_map}")

        if args.out_csv:
            csv_path = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv
            rows = []
            for sidx in sector_indices:
                sm = (emis == sidx) & domain
                totals, units = _summarize_pollutants(ds, sm, pollutants)
                for name in sorted(totals.keys()):
                    ta, tp, tt = totals[name]
                    rows.append(
                        {
                            "sector_index": sidx,
                            "sector": SECTOR_LABEL.get(sidx, str(sidx)),
                            "pollutant": name,
                            "area_kg_yr": ta,
                            "point_kg_yr": tp,
                            "total_kg_yr": tt,
                            "point_fraction": (tp / tt) if tt > 0 else float("nan"),
                            "units": units.get(name, ""),
                        }
                    )
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"CSV: {csv_path}")

        if args.out_points_csv:
            pcsv = args.out_points_csv if args.out_points_csv.is_absolute() else root / args.out_points_csv
            pm = base & (st == 2)
            idx = np.flatnonzero(pm)
            out_rows: list[dict[str, object]] = []
            for i in idx:
                row: dict[str, object] = {
                    "lon": float(lon[i]),
                    "lat": float(lat[i]),
                    "emission_category_index": int(emis[i]),
                    "sector": SECTOR_LABEL.get(int(emis[i]), str(int(emis[i]))),
                }
                for name in pollutants:
                    v = float(np.asarray(ds[name].values).ravel()[i])
                    row[name] = v
                out_rows.append(row)
            pd.DataFrame(out_rows).to_csv(pcsv, index=False)
            print(f"Point sources CSV ({len(out_rows)} rows): {pcsv}")
    finally:
        ds.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
