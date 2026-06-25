"""Interactive Folium map of CityChem point sources by SNAP sector."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from lines_osm import POLLUTANTS
from transform import _repo_root, load_config, resolve_path

SNAP_COLORS = {
    1: "#e6194b",
    2: "#3cb44b",
    3: "#4363d8",
    4: "#f58231",
    5: "#911eb4",
    6: "#42d4f4",
    7: "#f032e6",
    8: "#bfef45",
    9: "#fabed4",
    10: "#469990",
    11: "#dcbeff",
    12: "#9a6324",
}


def snap_label(cfg: dict, snap: int) -> str:
    raw = cfg["SNAP_TO_GNFR"][f"SNAP_{snap}"]
    if isinstance(raw, list):
        return " + ".join(raw)
    return str(raw)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4:
        return f"{v:,.0f}"
    if av >= 1:
        return f"{v:,.2f}"
    return f"{v:.4g}"


def _emission_rows(row: pd.Series) -> str:
    lines = []
    for pol in POLLUTANTS:
        val = float(row[pol])
        if val != 0:
            lines.append(f"{pol}: {_fmt(val)}")
    return "<br>".join(lines) if lines else "<i>all zero</i>"


def _jitter(lat: float, lon: float, snap: int, slot: int, meters: float = 35.0) -> tuple[float, float]:
    deg = meters / 111_320.0
    h = math.radians((snap * 47 + slot * 29) % 360)
    return lat + deg * math.sin(h), lon + deg * math.cos(h) / max(math.cos(math.radians(lat)), 0.2)


def _popup_html(title: str, xcor: float, ycor: float, body: str) -> str:
    return (
        f"<b>{html.escape(title)}</b><br>"
        f"xcor={xcor:.0f}, ycor={ycor:.0f}<br>{body}"
    )


def build_map(df: pd.DataFrame, cfg: dict, epsg: int) -> "folium.Map":
    try:
        import folium
    except ImportError as exc:
        raise ImportError("point_source_map needs folium: pip install folium") from exc

    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    df = df.copy()
    lons, lats = tr.transform(df["xcor"].astype(float), df["ycor"].astype(float))
    df["lon"] = lons
    df["lat"] = lats
    df["snap"] = df["snap"].astype(int)

    center = (float(df["lat"].mean()), float(df["lon"].mean()))
    fmap = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)

    coord_counts: dict[tuple[int, int], int] = {}
    for xcor, ycor in zip(df["xcor"], df["ycor"]):
        key = (int(xcor), int(ycor))
        coord_counts[key] = coord_counts.get(key, 0) + 1

    sector_layers: dict[int, folium.FeatureGroup] = {}
    for snap in sorted(df["snap"].unique()):
        color = SNAP_COLORS.get(snap, "#888888")
        label = snap_label(cfg, snap)
        sector_layers[snap] = folium.FeatureGroup(name=f"SNAP {snap} — {label}", show=True)

    combined = folium.FeatureGroup(name="All sectors (by location)", show=True)

    slot_at: dict[tuple[int, int, int], int] = {}
    for _, row in df.iterrows():
        snap = int(row["snap"])
        xcor, ycor = int(row["xcor"]), int(row["ycor"])
        lat, lon = float(row["lat"]), float(row["lon"])
        color = SNAP_COLORS.get(snap, "#888888")
        label = snap_label(cfg, snap)

        slot = slot_at.get((xcor, ycor, snap), 0)
        slot_at[(xcor, ycor, snap)] = slot + 1
        if coord_counts[(xcor, ycor)] > 1:
            mlat, mlon = _jitter(lat, lon, snap, slot)
        else:
            mlat, mlon = lat, lon

        popup = folium.Popup(
            _popup_html(f"SNAP {snap} — {label}", xcor, ycor, _emission_rows(row)),
            max_width=320,
        )
        folium.CircleMarker(
            location=[mlat, mlon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=popup,
        ).add_to(sector_layers[snap])

    for key, grp in sorted(sector_layers.items()):
        grp.add_to(fmap)

    for (xcor, ycor), sub in df.groupby(["xcor", "ycor"], sort=False):
        lat = float(sub["lat"].iloc[0])
        lon = float(sub["lon"].iloc[0])
        parts = []
        for _, r in sub.iterrows():
            snap = int(r["snap"])
            label = snap_label(cfg, snap)
            color = SNAP_COLORS.get(snap, "#888888")
            parts.append(
                f"<span style='color:{color}'>&#9679;</span> "
                f"<b>SNAP {snap}</b> ({html.escape(label)})<br>"
                f"<span style='margin-left:12px'>{_emission_rows(r)}</span>"
            )
        body = "<hr style='margin:6px 0'>".join(parts)
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color="#ffffff",
            fill=True,
            fill_color="#333333",
            fill_opacity=0.55,
            weight=2,
            popup=folium.Popup(
                _popup_html(f"Location ({len(sub)} sector(s))", float(xcor), float(ycor), body),
                max_width=360,
            ),
        ).add_to(combined)
    combined.add_to(fmap)

    table_rows = []
    for _, r in df.sort_values(["xcor", "ycor", "snap"]).iterrows():
        snap = int(r["snap"])
        color = SNAP_COLORS.get(snap, "#888888")
        pol_cells = "".join(f"<td style='text-align:right'>{_fmt(float(r[pol]))}</td>" for pol in POLLUTANTS)
        table_rows.append(
            f"<tr>"
            f"<td><span style='color:{color}'>&#9679;</span> {snap}</td>"
            f"<td>{html.escape(snap_label(cfg, snap))}</td>"
            f"<td>{int(r['xcor'])}</td><td>{int(r['ycor'])}</td>"
            f"{pol_cells}</tr>"
        )

    pol_headers = "".join(f"<th>{p}</th>" for p in POLLUTANTS)
    table_html = f"""
    <div id="ps-table-panel" style="
        position:fixed; top:12px; right:12px; z-index:9999; max-width:92vw;
        background:rgba(15,17,23,0.92); color:#e8eaf0; border:1px solid #3d4460;
        border-radius:8px; box-shadow:0 8px 28px rgba(0,0,0,0.55); display:none;
        max-height:70vh; overflow:auto; font-size:12px;">
      <div style="padding:10px 12px; border-bottom:1px solid #2a2f3d; font-weight:600;">
        Point sources — emissions (kg/yr)
      </div>
      <div style="overflow:auto; max-height:calc(70vh - 42px);">
        <table style="border-collapse:collapse; width:100%; white-space:nowrap;">
          <thead style="position:sticky; top:0; background:#181c25;">
            <tr>
              <th>SNAP</th><th>Sector</th><th>xcor</th><th>ycor</th>{pol_headers}
            </tr>
          </thead>
          <tbody>
            {"".join(table_rows)}
          </tbody>
        </table>
      </div>
    </div>
    <button id="ps-table-toggle" type="button" style="
        position:fixed; top:12px; right:12px; z-index:10000;
        background:#4f7cff; color:#0f1117; border:none; border-radius:6px;
        padding:8px 14px; font-size:12px; font-weight:600; cursor:pointer;
        box-shadow:0 4px 12px rgba(0,0,0,0.45);">
      Show emissions table
    </button>
    <script>
    (function() {{
      var btn = document.getElementById('ps-table-toggle');
      var panel = document.getElementById('ps-table-panel');
      btn.addEventListener('click', function() {{
        var open = panel.style.display !== 'none';
        panel.style.display = open ? 'none' : 'block';
        btn.textContent = open ? 'Show emissions table' : 'Hide emissions table';
        btn.style.right = open ? '12px' : 'calc(12px + min(92vw, 900px))';
      }});
    }})();
    </script>
    """

    legend_rows = []
    for snap in sorted(df["snap"].unique()):
        color = SNAP_COLORS.get(snap, "#888888")
        legend_rows.append(
            f"<span style='color:{color}'>&#9679;</span> SNAP {snap} — {html.escape(snap_label(cfg, snap))}"
        )
    legend_html = (
        "<div style='position:fixed; bottom:24px; left:12px; z-index:9999; "
        "background:rgba(15,17,23,0.92); color:#e8eaf0; padding:10px 12px; "
        "border:1px solid #3d4460; border-radius:8px; font-size:12px; max-width:320px;'>"
        "<b>SNAP sectors</b><br>"
        + "<br>".join(legend_rows)
        + "<br><br><span style='color:#aaa'>&#9679;</span> dark ring = combined location popup"
        + "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(table_html + legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    bounds = [[df["lat"].min(), df["lon"].min()], [df["lat"].max(), df["lon"].max()]]
    fmap.fit_bounds(bounds, padding=(30, 30))
    return fmap


def main():
    parser = argparse.ArgumentParser(description="Folium map of CityChem point sources by SNAP.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--csv", type=Path, default=None, help="point_source CSV (default from config)")
    parser.add_argument("--out", type=Path, default=None, help="output HTML path")
    args = parser.parse_args()

    root = _repo_root()
    cfg = load_config(args.config)
    epsg = int(cfg["EPSG"])
    city = str(cfg["City"])

    csv_path = args.csv or resolve_path(cfg["Output_folder"], root) / f"point_source_{city}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    out_path = args.out or csv_path.parent / f"point_source_{city}_map.html"

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"no rows in {csv_path}")

    fmap = build_map(df, cfg, epsg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
