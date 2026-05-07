"""Folium preview: CAMS↔facility link (two raster bands + polylines + markers).

Reads outputs from ``match-points`` (match CSV and optional 2-band link GeoTIFF on the
sector area-weight grid). First-class sector keys are listed in
:data:`PROXY.visualization.point_link_sectors.POINT_LINK_SECTOR_KEYS` (includes
``D_Fugitive``); any ``sector_key`` may be passed for titles and legends when invoking
this writer directly.
"""

from __future__ import annotations

import html as html_mod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.visualization.point_link_sectors import POINT_LINK_SECTOR_KEYS

from PROXY.visualization._mapbuilder import (
    add_raster_overlay,
    compute_view_context,
    create_folium_map_with_tiles,
    require_folium_imports,
    save_folium_map,
)
from PROXY.visualization.overlay_utils import read_band_warped_to_wgs84_grid, scalar_to_rgba

POINT_LINK_MARKER_POPUP_CSS = """
<style>
.plm-popup {
  font: 12px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1f2937;
  min-width: 200px;
  max-width: 360px;
}
.plm-popup .plm-title {
  font-size: 13px;
  font-weight: 650;
  color: #111827;
  margin: 0 0 6px 0;
  padding-bottom: 4px;
  border-bottom: 1px solid #e5e7eb;
}
.plm-popup .plm-tier {
  font-size: 12px;
  margin: 0 0 8px 0;
  color: #374151;
}
.plm-popup table.plm-kv {
  width: 100%;
  border-collapse: collapse;
  margin: 0;
}
.plm-popup table.plm-kv th,
.plm-popup table.plm-kv td {
  border-bottom: 1px solid #f1f5f9;
  padding: 3px 4px;
  text-align: left;
  font-size: 11.5px;
  vertical-align: top;
}
.plm-popup table.plm-kv th {
  color: #6b7280;
  font-weight: 600;
  width: 42%;
  padding-right: 8px;
}
.plm-popup code {
  background: #f3f4f6;
  padding: 1px 3px;
  border-radius: 2px;
  font-size: 11px;
}
</style>
""".strip()


def _fmt_scalar(val: Any) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        x = float(val)
        if abs(x - round(x)) < 1e-9 and abs(x) <= 1e12:
            return str(int(round(x)))
        if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-4):
            return f"{x:.3e}"
        return f"{x:.6g}"
    s = str(val).strip()
    return s[:500] + ("…" if len(s) > 500 else "")


def _cams_load_percentiles(loads: np.ndarray) -> tuple[float, float, float, float, int]:
    v = np.asarray(loads, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    p30, p60, p90 = (float(x) for x in np.percentile(v, [30.0, 60.0, 90.0]))
    mx = float(np.max(v))
    return p30, p60, p90, mx, n


def _tier_style(load: float, p30: float, p60: float, p90: float) -> tuple[str, int, str]:
    """Return ``(label, radius_px, stroke/fill_hex)`` for CAMS load percentiles (matched points only)."""
    x = float(load) if np.isfinite(load) else 0.0
    if x >= p90:
        return ("Very high emissions (90th–100th percentile)", 12, "#bf360c")
    if x >= p60:
        return ("High emissions (60th–90th percentile)", 9, "#e65100")
    if x >= p30:
        return ("Medium emissions (30th–60th percentile)", 7, "#f57c00")
    return ("Low emissions (0th–30th percentile)", 5, "#ff9800")


def _kv_table_html(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<th>{html_mod.escape(k)}</th>"
        f"<td>{v}</td>"
        "</tr>"
        for k, v in rows
        if v
    )
    if not body:
        return ""
    return f'<table class="plm-kv">{body}</table>'


def _cams_marker_popup_html(
    *,
    sector_key: str,
    tier_label: str,
    load: float,
    p30: float,
    p60: float,
    p90: float,
    vmax: float,
    rec: dict[str, Any],
) -> str:
    rows: list[tuple[str, str]] = []
    cid = rec.get("cams_point_id")
    if cid is not None and str(cid).strip() != "" and str(cid).lower() != "nan":
        rows.append(("CAMS point id", html_mod.escape(_fmt_scalar(cid))))
    load_note = (
        html_mod.escape(_fmt_scalar(load))
        + ' <span style="color:#6b7280">(match CSV; same variable as match-points / CAMS NetCDF)</span>'
    )
    rows.append(("CAMS pollutant load", load_note))
    rows.append(
        (
            "Percentile thresholds (this file)",
            html_mod.escape(
                f"P30={_fmt_scalar(p30)}, P60={_fmt_scalar(p60)}, P90={_fmt_scalar(p90)}, max={_fmt_scalar(vmax)}"
            ),
        )
    )
    lat, lon = rec.get("cams_latitude"), rec.get("cams_longitude")
    if lat is not None and lon is not None:
        rows.append(
            (
                "Coordinates",
                html_mod.escape(f"{float(lat):.5f} N, {float(lon):.5f} E"),
            )
        )
    fn = rec.get("facility_name")
    fi = rec.get("facility_id")
    if fn is not None or fi is not None:
        link_bits = " ".join(
            x for x in (html_mod.escape(_fmt_scalar(fn)), html_mod.escape(_fmt_scalar(fi))) if x
        )
        if link_bits.strip():
            rows.append(("Matched facility", link_bits))
    mid = rec.get("match_id")
    if mid is not None and str(mid).strip():
        rows.append(("Match id", html_mod.escape(_fmt_scalar(mid))))
    inner = _kv_table_html(rows)
    return (
        '<div class="plm-popup">'
        f'<div class="plm-title">{html_mod.escape(sector_key)} &mdash; CAMS source</div>'
        f'<div class="plm-tier">{html_mod.escape(tier_label)}</div>'
        f"{inner}"
        "</div>"
    )


# Keys omitted from facility popups (match diagnostics; CAMS-side info shown on orange markers).
_FACILITY_POPUP_SUPPRESS: frozenset[str] = frozenset(
    {
        "cams_latitude",
        "cams_longitude",
        "facility_latitude",
        "facility_longitude",
        "distance_km",
        "match_stage",
        "score",
        "score_distance",
        "score_pollutant",
        "score_activity",
        "cams_point_id",
        "cams_pollutant_value",
        "match_id",
        "sector",
        "year",
        "fallback_score",
        "match_method",
    }
)


def _facility_marker_popup_html(*, sector_key: str, rec: dict[str, Any]) -> str:
    priority_keys = [
        ("Facility name", "facility_name"),
        ("Facility id", "facility_id"),
        ("Dataset pollutant (scoring)", "eprtr_pollutant"),
        ("Reporting year", "reporting_year"),
        ("Sector code", "eprtr_sector_code"),
        ("Registry", "registry"),
        ("Source dataset", "source_dataset"),
        ("Coordinates", "_coord_pair"),
    ]
    rows: list[tuple[str, str]] = []
    lat, lon = rec.get("facility_latitude"), rec.get("facility_longitude")
    coord_html = ""
    if lat is not None and lon is not None:
        try:
            coord_html = html_mod.escape(f"{float(lat):.5f} N, {float(lon):.5f} E")
        except (TypeError, ValueError):
            coord_html = ""
    for label, key in priority_keys:
        if key == "_coord_pair":
            if coord_html:
                rows.append((label, coord_html))
            continue
        if key not in rec:
            continue
        raw = rec.get(key)
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            continue
        if isinstance(raw, str) and not raw.strip():
            continue
        disp = html_mod.escape(_fmt_scalar(raw))
        rows.append((label, disp))

    done = {k for _, k in priority_keys if not k.startswith("_")}
    for k in sorted(rec.keys()):
        if k in done or k in _FACILITY_POPUP_SUPPRESS:
            continue
        raw = rec.get(k)
        if raw is None:
            continue
        if isinstance(raw, (dict, list)):
            continue
        if isinstance(raw, str) and not raw.strip():
            continue
        if isinstance(raw, float) and not np.isfinite(raw):
            continue
        label = k.replace("_", " ")
        rows.append((label, html_mod.escape(_fmt_scalar(raw))))

    inner = _kv_table_html(rows)
    return (
        '<div class="plm-popup">'
        f'<div class="plm-title">{html_mod.escape(sector_key)} &mdash; Facility (dataset)</div>'
        f"{inner}"
        "</div>"
    )


def write_point_link_context_html(
    *,
    root: Path,
    sector_key: str,
    link_tif: Path,
    matches_csv: Path,
    out_html: Path,
    pad_deg: float,
    max_width: int,
    max_height: int,
    region: str | None = None,
    override_bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write ``{sector}_point_context_map.html`` with two toggled raster layers and link vectors."""
    need = require_folium_imports()
    folium = need["folium"]

    view = compute_view_context(
        link_tif,
        pad_deg=pad_deg,
        max_width=max_width,
        max_height=max_height,
        override_bbox=override_bbox,
        region=region,
    )

    z1, _nd1 = read_band_warped_to_wgs84_grid(
        link_tif,
        band=1,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        width=view.gw,
        height=view.gh,
    )
    z2, _nd2 = read_band_warped_to_wgs84_grid(
        link_tif,
        band=2,
        west=view.west,
        south=view.south,
        east=view.east,
        north=view.north,
        width=view.gw,
        height=view.gh,
    )

    rgba1 = scalar_to_rgba(
        z1,
        colour_mode="log",
        cmap_name="Oranges",
        hide_zero=True,
        nodata_val=None,
    )
    rgba2 = scalar_to_rgba(
        z2,
        colour_mode="log",
        cmap_name="Blues",
        hide_zero=True,
        nodata_val=None,
    )

    fmap = create_folium_map_with_tiles(view, zoom_start=8, default_basemap="satellite")
    add_raster_overlay(
        fmap,
        rgba1,
        view,
        name="Band 1 — CAMS point mass (on ref grid)",
        opacity=0.72,
        show=True,
    )
    add_raster_overlay(
        fmap,
        rgba2,
        view,
        name="Band 2 — linked facility mass (on ref grid)",
        opacity=0.72,
        show=False,
    )

    df = pd.read_csv(matches_csv)
    req = {"cams_latitude", "cams_longitude", "facility_latitude", "facility_longitude"}
    if not req.issubset(df.columns):
        raise ValueError(
            f"{matches_csv.name} missing columns {sorted(req)}; re-run match-points to refresh."
        )

    if "cams_pollutant_value" not in df.columns:
        df = df.copy()
        df["cams_pollutant_value"] = 0.0
    loads = pd.to_numeric(df["cams_pollutant_value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    p30, p60, p90, vmax, n_matched = _cams_load_percentiles(loads)
    pct_note = (
        f"CAMS marker size and colour use <code>cams_pollutant_value</code> from the match file "
        f"({n_matched} point(s)). Percentiles are computed over those values only "
        f"(not the full CAMS grid). Maximum in file: <code>{html_mod.escape(_fmt_scalar(vmax))}</code>."
    )

    fg_links = folium.FeatureGroup(name="CAMS to facility links", show=True)
    fg_cams = folium.FeatureGroup(name="CAMS points (size = emission tier)", show=True)
    fg_fac = folium.FeatureGroup(name="Linked facilities", show=True)

    if not df.empty:
        fmap.get_root().header.add_child(folium.Element(POINT_LINK_MARKER_POPUP_CSS))

    for rec in df.to_dict(orient="records"):
        clat = float(rec["cams_latitude"])
        clon = float(rec["cams_longitude"])
        flat = float(rec["facility_latitude"])
        flon = float(rec["facility_longitude"])
        load = float(rec.get("cams_pollutant_value") or 0.0)
        tier_label, cam_r, cam_color = _tier_style(load, p30, p60, p90)
        name_esc = html_mod.escape(_fmt_scalar(rec.get("facility_name"))[:120])
        fid_esc = html_mod.escape(_fmt_scalar(rec.get("facility_id"))[:120])
        cams_html = _cams_marker_popup_html(
            sector_key=sector_key,
            tier_label=tier_label,
            load=load,
            p30=p30,
            p60=p60,
            p90=p90,
            vmax=vmax,
            rec=rec,
        )
        fac_html = _facility_marker_popup_html(sector_key=sector_key, rec=rec)
        link_tt = f"{name_esc} ({fid_esc}) &mdash; click markers for details"
        folium.PolyLine(
            locations=[(clat, clon), (flat, flon)],
            color="#c62828",
            weight=2,
            opacity=0.65,
            tooltip=folium.Tooltip(link_tt, sticky=True),
        ).add_to(fg_links)
        folium.CircleMarker(
            location=(clat, clon),
            radius=cam_r,
            color=cam_color,
            fill=True,
            fill_color=cam_color,
            fill_opacity=0.88,
            weight=1,
            popup=folium.Popup(cams_html, max_width=380),
            tooltip=folium.Tooltip(html_mod.escape(tier_label), sticky=True),
        ).add_to(fg_cams)
        folium.CircleMarker(
            location=(flat, flon),
            radius=6,
            color="#0d47a1",
            fill=True,
            fill_color="#1565c0",
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(fac_html, max_width=380),
            tooltip=folium.Tooltip(f"{name_esc} ({fid_esc})", sticky=True),
        ).add_to(fg_fac)

    fg_links.add_to(fmap)
    fg_cams.add_to(fmap)
    fg_fac.add_to(fmap)

    tier_legend = (
        "<p style=\"margin:8px 0 4px 0;font-weight:650;\">CAMS points (orange markers)</p>"
        "<p style=\"margin:0 0 6px 0;font-size:12px;color:#374151;\">"
        "Marker <b>radius</b> scales with total CAMS pollutant load at the source, "
        "using four bands by <b>percentile within this match file</b>:</p>"
        "<ul style=\"margin:0 0 8px 0;padding-left:18px;font-size:12px;line-height:1.45;\">"
        "<li><span style=\"color:#bf360c;font-weight:600;\">Very high</span> "
        "&mdash; 90th&ndash;100th percentile (top 10%)</li>"
        "<li><span style=\"color:#e65100;font-weight:600;\">High</span> "
        "&mdash; 60th&ndash;90th percentile</li>"
        "<li><span style=\"color:#f57c00;font-weight:600;\">Medium</span> "
        "&mdash; 30th&ndash;60th percentile</li>"
        "<li><span style=\"color:#ff9800;font-weight:600;\">Low</span> "
        "&mdash; 0th&ndash;30th percentile</li>"
        "</ul>"
        "<p style=\"margin:0 0 6px 0;font-size:11.5px;color:#4b5563;\">"
        f"{pct_note}</p>"
        "<p style=\"margin:0 0 2px 0;font-size:12px;\">"
        "<b>Blue</b> markers are dataset facilities. <b>Click</b> orange or blue markers for "
        "CAMS load, percentile context, and facility / match metadata (all columns from the match CSV).</p>"
    )

    sectors_note = (
        '<p style="margin:8px 0 0 0;font-size:11px;color:#4b5563;">'
        "Sectors with point-link matching in PROXY: "
        f"<code>{html_mod.escape(', '.join(sorted(POINT_LINK_SECTOR_KEYS)))}</code>."
        "</p>"
    )

    legend = (
        "<div class=\"pl-legend\" style=\"position:fixed;top:12px;right:12px;z-index:9999;"
        "max-width:340px;padding:10px 12px;background:rgba(255,255,255,0.94);"
        "border-radius:6px;font:13px/1.35 system-ui,sans-serif;box-shadow:0 1px 6px rgba(0,0,0,0.2);\">"
        f"<p style=\"margin:0 0 6px 0;font-weight:650;\">{html_mod.escape(sector_key)} point proxy</p>"
        "<p style=\"margin:0 0 6px 0;\">Two GeoTIFF bands (same CRS as the sector area weights) plus "
        "links in map space (straight lines in WGS84). Matching is <b>one-to-one</b> (each facility id "
        "once) with a <b>maximum distance</b> from ``eprtr_scoring.yaml`` / sector YAML "
        "(see match JSON <code>max_match_distance_km</code> and <code>preferred_eprtr_sector_codes</code>). "
        "CAMS points are processed by descending "
        "load so each takes the best eligible facility still available.</p>"
        f"{tier_legend}"
        "<ul style=\"margin:0;padding-left:18px;\">"
        "<li><b>Orange</b> raster: mass summed at pixels under CAMS coordinates.</li>"
        "<li><b>Blue</b> raster: same mass summed at pixels under matched facility coordinates.</li>"
        "<li><b>Red lines</b>: CAMS point to linked facility.</li>"
        "</ul>"
        f"{sectors_note}"
        "</div>"
    )

    return save_folium_map(fmap, out_html, root, legend_html=legend)
