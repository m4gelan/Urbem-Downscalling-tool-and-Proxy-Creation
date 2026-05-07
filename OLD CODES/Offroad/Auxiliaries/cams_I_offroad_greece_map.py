#!/usr/bin/env python3
"""
Interactive Folium map of CAMS-REG-ANT GNFR I (Off-road) for Greece.

Area sources (source_type_index == 1) are drawn as grid-cell rectangles; point sources
(== 2) as circle markers. Basemaps: OpenStreetMap (default), light (CartoDB positron),
and Esri World Imagery (satellite). Use the layer control to switch.

emission_category_index: I = 12 (1-based) in this CAMS file (F is split into F1..F4).

Usage (from project root):
  python Offroad/Auxiliaries/cams_I_offroad_greece_map.py
  python Offroad/Auxiliaries/cams_I_offroad_greece_map.py --pollutant pm2_5
  python Offroad/Auxiliaries/cams_I_offroad_greece_map.py --out-map Offroad/outputs/cams_I_offroad_GRC_nox.html

Requires: xarray, netCDF4, numpy, pandas; folium, branca
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# GNFR I = index 12 (1-based) in this CAMS file (because F is split into F1..F4)
IDX_I_OFFROAD = 12

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
        raise SystemExit(
            f"Country {iso3!r} not in NetCDF country_id (have {len(codes)} countries)."
        ) from exc


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


def _print_table(
    totals: dict[str, tuple[float, float, float]],
    units: dict[str, str],
) -> None:
    print("Totals by pollutant: area vs point (GNFR I, same domain)")
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


def _add_extra_basemaps(fmap) -> None:
    import folium

    folium.TileLayer("CartoDB positron", name="Map (light)", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(fmap)


def add_cams_i_layers(
    fmap,
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    *,
    map_max_area_cells: int = 0,
    map_sample_seed: int = 42,
) -> None:
    import folium
    from branca.colormap import StepColormap

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    vals = np.asarray(ds[pollutant].values).ravel().astype(np.float64)
    m_area = base_mask & (st == 1)
    m_point = base_mask & (st == 2)

    pos = vals > 0
    vpos = vals[pos & (m_area | m_point)]
    if vpos.size:
        vmin = float(np.min(vpos))
        vmax = float(np.max(vpos))
        if vmin >= vmax:
            vmax = vmin * 1.001 if vmin > 0 else 1.0
    else:
        vmin, vmax = 0.0, 1.0

    q = np.quantile(vpos, np.linspace(0.0, 1.0, 11)) if vpos.size else np.linspace(vmin, vmax, 11)
    q = np.asarray(q, dtype=float)
    for i in range(1, q.size):
        if not (q[i] > q[i - 1]):
            q[i] = q[i - 1] + max(1e-12, abs(q[i - 1]) * 1e-12)

    colors = [
        "#2c7bb6",
        "#00a6ca",
        "#00ccbc",
        "#90eb9d",
        "#ffff8c",
        "#f9d057",
        "#f29e2e",
        "#e76818",
        "#d7191c",
        "#7f0000",
    ]
    cmap = StepColormap(
        colors=colors,
        index=q.tolist(),
        vmin=float(q[0]),
        vmax=float(q[-1]),
        caption=f"{pollutant} (kg/year) — deciles",
    )

    def radius_for(v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            return 3.0
        t = math.sqrt(v / max(vmax, 1e-30))
        return min(28.0, 4.0 + 18.0 * t)

    fg_area = folium.FeatureGroup(name="I OffRoad - area (grid cell)", show=True)
    fg_point = folium.FeatureGroup(name="I OffRoad - point", show=True)

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
        idx_area = np.sort(
            rng.choice(idx_area, size=map_max_area_cells, replace=False)
        )
        print(
            f"Map: drawing {map_max_area_cells} of {n_area_all} area grid cells "
            f"(totals in table use full domain)."
        )

    for i in idx_area:
        v = float(vals[i])
        if not math.isfinite(v):
            continue
        color = cmap(v) if v > 0 else "#cccccc"
        li, ji = int(lon_ii[i]), int(lat_ii[i])
        west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
        south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
        if south > north:
            south, north = north, south
        if west > east:
            west, east = east, west
        pop = (
            f"<b>{html.escape(pollutant)}</b><br/>{v:.6g} kg/yr<br/>area (grid cell)<br/>"
            f"cell lon [{west:.4f}, {east:.4f}]<br/>"
            f"cell lat [{south:.4f}, {north:.4f}]<br/>"
            f"centre lon {lon[i]:.4f}, lat {lat[i]:.4f}"
        )
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#333333",
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
        color = cmap(v) if v > 0 else "#cccccc"
        r = radius_for(v)
        folium.CircleMarker(
            location=[float(lat[i]), float(lon[i])],
            radius=r,
            color="#333333",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{html.escape(pollutant)}</b><br/>{v:.6g} kg/yr<br/>point<br/>"
                f"lon {lon[i]:.4f}, lat {lat[i]:.4f}",
                max_width=260,
            ),
        ).add_to(fg_point)

    fg_area.add_to(fmap)
    fg_point.add_to(fmap)
    cmap.add_to(fmap)


def build_folium_map(
    ds,
    base_mask: np.ndarray,
    pollutant: str,
    *,
    out_html: Path,
    map_center_latlon: tuple[float, float] | None,
    zoom_start: int,
    map_max_area_cells: int,
    map_sample_seed: int,
) -> None:
    import folium

    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    if map_center_latlon is not None:
        center_lat, center_lon = map_center_latlon
    else:
        center_lat = float(np.mean(lat[base_mask]))
        center_lon = float(np.mean(lon[base_mask]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
    )
    _add_extra_basemaps(fmap)
    add_cams_i_layers(
        fmap,
        ds,
        base_mask,
        pollutant,
        map_max_area_cells=map_max_area_cells,
        map_sample_seed=map_sample_seed,
    )
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="CAMS GNFR I (Off-road): map + area/point totals."
    )
    ap.add_argument("--nc", type=Path, default=DEFAULT_NC, help="CAMS NetCDF path")
    ap.add_argument("--country", default="GRC", help="ISO3 (default GRC)")
    ap.add_argument(
        "--all-countries",
        action="store_true",
        help="Ignore --country; use full domain (still GNFR I only)",
    )
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional WGS84 bbox",
    )
    ap.add_argument("--pollutant", default="nox", help="Variable for map colouring")
    ap.add_argument("--no-map", action="store_true", help="Skip HTML output")
    ap.add_argument(
        "--no-table",
        action="store_true",
        help="Do not print area/point totals table",
    )
    ap.add_argument(
        "--out-map",
        type=Path,
        default=None,
        help="Output HTML (default: Offroad/outputs/cams_I_offroad_<ISO>_pollutant.html)",
    )
    ap.add_argument(
        "--map-max-area-cells",
        type=int,
        default=25_000,
        help="Max area rectangles on map (0 = all)",
    )
    ap.add_argument("--map-sample-seed", type=int, default=42)
    ap.add_argument("--map-zoom", type=int, default=None)
    ap.add_argument("--map-center-lat", type=float, default=None)
    ap.add_argument("--map-center-lon", type=float, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    nc_path = args.nc if args.nc.is_absolute() else root / args.nc
    if not nc_path.is_file():
        raise SystemExit(f"NetCDF not found: {nc_path}")

    import xarray as xr

    ds = xr.open_dataset(nc_path)
    try:
        pollutants = _pollutant_vars(ds)
        if not pollutants:
            raise SystemExit("No pollutant variables with dimension 'source' found.")

        country_1b: int | None = None if args.all_countries else _country_index_1based(
            ds, args.country
        )

        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel()
        lat = np.asarray(ds["latitude_source"].values).ravel()
        bbox_t = tuple(args.bbox) if args.bbox else None
        sector_i = emis == IDX_I_OFFROAD
        domain = _build_domain_mask(lon, lat, ci, country_1b, bbox_t)
        base = sector_i & domain
        if int(np.count_nonzero(base)) == 0:
            raise SystemExit("No GNFR I sources match country/bbox filter.")

        print(f"NetCDF: {nc_path.name}")
        print(
            f"Filter: GNFR I (emission_category_index={IDX_I_OFFROAD}), "
            f"{'all countries' if args.all_countries else args.country.upper()}"
        )
        if args.bbox:
            print(
                f"  bbox lon[{args.bbox[0]},{args.bbox[2]}] "
                f"lat[{args.bbox[1]},{args.bbox[3]}]"
            )
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        n_area = int(np.count_nonzero(base & (st == 1)))
        n_point = int(np.count_nonzero(base & (st == 2)))
        print(f"Sources in domain: {int(np.count_nonzero(base))} (area {n_area}, point {n_point})")
        print()

        totals, units = _summarize_pollutants(ds, base, pollutants)
        if not args.no_table:
            _print_table(totals, units)
            print()

        pol = args.pollutant.strip().lower()
        if pol not in totals:
            raise SystemExit(
                f"Unknown pollutant {pol!r}. Examples: {', '.join(pollutants[:8])}"
            )

        if not args.no_map:
            out_map = args.out_map
            tag = "ALL" if args.all_countries else args.country.upper()
            if out_map is None:
                out_map = (
                    root / "Offroad" / "outputs" / f"cams_I_offroad_{tag}_{pol}.html"
                )
            else:
                out_map = out_map if out_map.is_absolute() else root / out_map

            mclat, mclon = args.map_center_lat, args.map_center_lon
            if (mclat is None) ^ (mclon is None):
                raise SystemExit("Set both --map-center-lat and --map-center-lon, or neither.")
            map_center: tuple[float, float] | None
            if mclat is not None:
                map_center = (float(mclat), float(mclon))
            elif args.all_countries:
                map_center = (51.0, 14.0)
            else:
                map_center = None

            zoom_use = args.map_zoom if args.map_zoom is not None else (
                4 if args.all_countries else 7
            )
            build_folium_map(
                ds,
                base,
                pol,
                out_html=out_map,
                map_center_latlon=map_center,
                zoom_start=int(zoom_use),
                map_max_area_cells=max(0, int(args.map_max_area_cells)),
                map_sample_seed=int(args.map_sample_seed),
            )
            print(f"Interactive map ({pol}): {out_map}")

        if args.out_csv:
            csv_path = args.out_csv if args.out_csv.is_absolute() else root / args.out_csv
            rows = []
            for name in sorted(totals.keys()):
                ta, tp, tt = totals[name]
                rows.append(
                    {
                        "pollutant": name,
                        "area_kg_yr": ta,
                        "point_kg_yr": tp,
                        "total_kg_yr": tt,
                        "point_fraction": (tp / tt) if tt > 0 else float("nan"),
                        "units": units.get(name, ""),
                    }
                )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"CSV: {csv_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
