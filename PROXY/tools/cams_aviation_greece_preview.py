"""Interactive Folium map: CAMS-REG GNFR H (aviation) point sources for Greece.

Reads the CAMS NetCDF used elsewhere in PROXY (paths.yaml ``emissions.cams_2019_nc``).
Use this to visually check whether CAMS airport coordinates match reality before
building ``aviation_pointsource.tif`` / airport matching.

Usage (from project root):
  python PROXY/tools/cams_aviation_greece_preview.py
  python PROXY/tools/cams_aviation_greece_preview.py --out Output/maps/cams_H_GRC_points.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

_root_boot = Path(__file__).resolve().parents[2]
if str(_root_boot) not in sys.path:
    sys.path.insert(0, str(_root_boot))

import numpy as np
import xarray as xr
import yaml

from PROXY.core.cams.domain import country_index_1based
from PROXY.core.cams.gnfr import gnfr_code_to_index
from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions


def _escape(s: object) -> str:
    return html.escape(str(s), quote=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML (default: Output/maps/cams_H_aviation_GRC_2019.html under root).",
    )
    ap.add_argument("--country", type=str, default="GRC", help="ISO3 country_id in CAMS.")
    ap.add_argument(
        "--include-area-rows",
        action="store_true",
        help="Also plot GNFR H rows with source_type_index=1 (area); Greece has none in v8.1 2019.",
    )
    args = ap.parse_args()
    root = args.root.resolve()

    try:
        import folium
    except ImportError:
        print("Install folium: pip install folium", file=sys.stderr)
        return 1

    paths_yaml = root / "PROXY" / "config" / "paths.yaml"
    if not paths_yaml.is_file():
        print(f"paths.yaml not found: {paths_yaml}", file=sys.stderr)
        return 1
    with paths_yaml.open(encoding="utf-8") as f:
        path_cfg = yaml.safe_load(f)
    em = (path_cfg or {}).get("emissions") or {}
    nc_rel = em.get("cams_2019_nc")
    if not nc_rel:
        print("paths.yaml missing emissions.cams_2019_nc", file=sys.stderr)
        return 1
    nc_path = discover_cams_emissions(root, resolve_path(root, Path(str(nc_rel))))

    out = args.out
    if out is None:
        out = root / "Output" / "maps" / "cams_H_aviation_GRC_2019.html"
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    iso3 = str(args.country).strip().upper()
    gnfr_h = gnfr_code_to_index("H")

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    try:
        cidx = country_index_1based(ds, iso3)
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(ds["latitude_source"].values).ravel().astype(np.float64)

        base = (ci == cidx) & (emis == gnfr_h)
        m_point = base & (st == 2)
        m_area = base & (st == 1)
        idx_point = np.where(m_point)[0]
        idx_area = np.where(m_area)[0] if args.include_area_rows else np.array([], dtype=np.int64)

        pols = ["nox", "co", "nmvoc", "sox", "nh3", "pm2_5", "pm10", "ch4"]
        series: dict[str, np.ndarray] = {}
        for p in pols:
            if p in ds:
                series[p] = np.asarray(ds[p].values).ravel().astype(np.float64)

        # Greece overview
        lat_c = 39.07
        lon_c = 22.43
        fmap = folium.Map(
            location=[lat_c, lon_c],
            zoom_start=7,
            tiles=None,
            control_scale=True,
        )
        folium.TileLayer(
            "CartoDB positron",
            name="Light (CartoDB Positron)",
            control=True,
            show=True,
        ).add_to(fmap)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri",
            name="Satellite (Esri)",
            max_zoom=19,
            control=True,
            show=False,
        ).add_to(fmap)

        fg_p = folium.FeatureGroup(name=f"CAMS GNFR H point ({iso3}, n={len(idx_point)})", show=True)
        for i in idx_point:
            parts = [
                f"<b>CAMS GNFR H point</b> {_escape(iso3)}",
                f"idx={int(i)}",
                f"lon={lon[i]:.5f}, lat={lat[i]:.5f}",
            ]
            for p in pols:
                if p not in series:
                    continue
                v = float(series[p][i])
                if np.isfinite(v) and v != 0.0:
                    parts.append(f"{p}={v:.6g}")
            folium.CircleMarker(
                location=[float(lat[i]), float(lon[i])],
                radius=6,
                color="#1565c0",
                weight=2,
                fill=True,
                fill_color="#42a5f5",
                fill_opacity=0.85,
                popup=folium.Popup("<br/>".join(parts), max_width=320),
            ).add_to(fg_p)
        fg_p.add_to(fmap)

        if args.include_area_rows and len(idx_area):
            fg_a = folium.FeatureGroup(
                name=f"CAMS GNFR H area ({iso3}, n={len(idx_area)})", show=True
            )
            for i in idx_area:
                parts = [
                    f"<b>CAMS GNFR H area</b> {_escape(iso3)}",
                    f"idx={int(i)}",
                    f"lon={lon[i]:.5f}, lat={lat[i]:.5f}",
                ]
                for p in pols:
                    if p not in series:
                        continue
                    v = float(series[p][i])
                    if np.isfinite(v) and v != 0.0:
                        parts.append(f"{p}={v:.6g}")
                folium.CircleMarker(
                    location=[float(lat[i]), float(lon[i])],
                    radius=8,
                    color="#c62828",
                    weight=2,
                    fill=True,
                    fill_color="#ef5350",
                    fill_opacity=0.7,
                    popup=folium.Popup("<br/>".join(parts), max_width=320),
                ).add_to(fg_a)
            fg_a.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        legend = (
            f"<div style='position: fixed; top: 10px; left: 50px; width: 420px; height: auto; "
            f"z-index: 9999; font-size: 13px; background-color: rgba(255,255,255,0.92); "
            f"padding: 10px; border: 1px solid #ccc; border-radius: 4px;'>"
            f"<b>CAMS-REG v8.1 (2019)</b> — GNFR <b>H</b> (aviation), "
            f"<code>country_id={_escape(iso3)}</code><br/>"
            f"Point sources: <b>{len(idx_point)}</b>"
            f"{'' if not args.include_area_rows else f'; area rows shown: <b>{len(idx_area)}</b>'}"
            f"<br/><span style='font-size:11px;color:#555'>NetCDF: {_escape(nc_path.name)}</span>"
            f"</div>"
        )
        fmap.get_root().html.add_child(folium.Element(legend))

        fmap.fit_bounds([[34.75, 19.1], [41.85, 29.0]])
        fmap.save(str(out))
    finally:
        ds.close()

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
