from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd


from proxy.writers.point_link import _facility_links


def _jitter_latlon(uid: str, lat: float, lon: float, meters: float = 180.0) -> tuple[float, float]:
    deg = meters / 111_320.0
    h = sum(ord(c) for c in str(uid)) % 360
    rad = math.radians(h)
    return lat + deg * math.sin(rad), lon + deg * math.cos(rad)


def _in_bbox_wgs84(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def _facility_info_for_line(m: dict[str, Any]) -> dict[str, Any] | None:
    links = _facility_links(m)
    if links:
        info = links[0].get("facility_info") or {}
        if info.get("lon") is not None and info.get("lat") is not None:
            return info
    for key in (
        "corine_facility_info",
        "uwwtd_facility_info",
        "eprtr_point_info",
        "jrc_point_info",
        "osm_facility_info",
        "riurbans_point_info",
    ):
        info = m.get(key)
        if isinstance(info, dict) and info.get("lon") is not None and info.get("lat") is not None:
            return info
    return None


def write_point_match_map(
    matches: dict[int, dict[str, Any]],
    jrc_points: dict[str, dict[str, Any]],
    out_html: Path,
    *,
    eprtr_points: dict[str, dict[str, Any]] | None = None,
    osm_facility_points: dict[str, dict[str, Any]] | None = None,
    osm_polygons_gdf: gpd.GeoDataFrame | None = None,
    corine_facility_points: dict[str, dict[str, Any]] | None = None,
    uwwtd_facility_points: dict[str, dict[str, Any]] | None = None,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> Path:
    """
    Interactive map: CAMS (green/red), optional JRC (blue/purple), EPRTR (cyan/orange),
    optional OSM aerodrome match points (indigo / violet), optional OSM polygon outlines,
    optional CORINE L124 patch points (brown / tan), optional UWWTD plants (teal).
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError("Point-match map needs folium: pip install folium") from exc

    eprtr_points = eprtr_points or {}
    osm_facility_points = osm_facility_points or {}
    corine_facility_points = corine_facility_points or {}
    uwwtd_facility_points = uwwtd_facility_points or {}

    assigned_jrc: set[str] = set()
    jrc_to_cams: dict[str, list[tuple[int, str, float]]] = {}
    assigned_eprtr: set[str] = set()
    eprtr_to_cams: dict[str, list[tuple[int, str, float]]] = {}
    assigned_osm: set[str] = set()
    osm_to_cams: dict[str, list[tuple[int, str, float]]] = {}
    assigned_corine: set[str] = set()
    corine_to_cams: dict[str, list[tuple[int, str, float]]] = {}
    assigned_uwwtd: set[str] = set()
    uwwtd_to_cams: dict[str, list[tuple[int, str, float]]] = {}

    assigned_ri: set[str] = set()
    ri_to_cams: dict[str, list[tuple[int, str, float]]] = {}

    for pid, m in matches.items():
        dist = float(m["scoring_value"]) if m.get("scoring_value") is not None else float("nan")
        matched_yes = m.get("matched") == "yes"
        for lk in _facility_links(m):
            fid = str(lk.get("facility_id") or "")
            d = float(lk.get("scoring_value") or dist)
            src = str(m.get("match_source") or "")
            if fid and (src == "riurbans" or fid.startswith("ri_")):
                ri_to_cams.setdefault(fid, []).append((int(pid), str(m.get("matched", "")), d))
                if matched_yes:
                    assigned_ri.add(fid)
        jid = m.get("jrc_point_id")
        if jid:
            jid = str(jid)
            jrc_to_cams.setdefault(jid, []).append((int(pid), str(m.get("matched", "")), dist))
            if matched_yes:
                assigned_jrc.add(jid)
        eid = m.get("eprtr_point_id")
        if eid:
            eid = str(eid)
            eprtr_to_cams.setdefault(eid, []).append((int(pid), str(m.get("matched", "")), dist))
            if matched_yes:
                assigned_eprtr.add(eid)
        oid = m.get("osm_facility_id")
        if oid:
            oid = str(oid)
            osm_to_cams.setdefault(oid, []).append((int(pid), str(m.get("matched", "")), dist))
            if matched_yes:
                assigned_osm.add(oid)
        cid = m.get("corine_facility_id")
        if cid:
            cid = str(cid)
            corine_to_cams.setdefault(cid, []).append((int(pid), str(m.get("matched", "")), dist))
            if matched_yes and m.get("match_source") == "corine":
                assigned_corine.add(cid)
        wid = m.get("uwwtd_facility_id")
        if wid:
            wid = str(wid)
            uwwtd_to_cams.setdefault(wid, []).append((int(pid), str(m.get("matched", "")), dist))
            if matched_yes:
                assigned_uwwtd.add(wid)

    lats: list[float] = []
    lons: list[float] = []
    for m in matches.values():
        c = m["cams"]
        lats.append(float(c["latitude"]))
        lons.append(float(c["longitude"]))
    for j in jrc_points.values():
        lats.append(float(j["lat"]))
        lons.append(float(j["lon"]))
    for e in eprtr_points.values():
        lats.append(float(e["lat"]))
        lons.append(float(e["lon"]))
    for o in osm_facility_points.values():
        lats.append(float(o["lat"]))
        lons.append(float(o["lon"]))
    for cf in corine_facility_points.values():
        lats.append(float(cf["lat"]))
        lons.append(float(cf["lon"]))
    for uw in uwwtd_facility_points.values():
        lats.append(float(uw["lat"]))
        lons.append(float(uw["lon"]))
    if osm_polygons_gdf is not None and not osm_polygons_gdf.empty:
        b = osm_polygons_gdf.to_crs("EPSG:4326").total_bounds
        w, s, e, n = (float(x) for x in b)
        lons.extend([w, e, w, e])
        lats.extend([s, s, n, n])
    if not lats:
        raise ValueError("No coordinates to plot")

    if bbox_wgs84 is not None:
        w, s, e, n = bbox_wgs84
        center = ((s + n) / 2.0, (w + e) / 2.0)
        zoom = 10
    else:
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
        zoom = 7
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    if bbox_wgs84 is not None:
        fmap.fit_bounds([[s, w], [n, e]])

    if osm_polygons_gdf is not None and not osm_polygons_gdf.empty:
        g_osm = osm_polygons_gdf.to_crs("EPSG:4326")
        folium.GeoJson(
            data=g_osm.to_json(),
            name="OSM polygons (GPKG, clipped)",
            style_function=lambda _feat: {
                "fillColor": "#42a5f5",
                "color": "#0d47a1",
                "weight": 1.5,
                "fillOpacity": 0.18,
            },
        ).add_to(fmap)

    def _show_cams(m: dict[str, Any]) -> bool:
        if bbox_wgs84 is None:
            return True
        c = m["cams"]
        clon, clat = float(c["longitude"]), float(c["latitude"])
        if _in_bbox_wgs84(clon, clat, bbox_wgs84):
            return True
        info = _facility_info_for_line(m)
        if info is not None:
            return _in_bbox_wgs84(float(info["lon"]), float(info["lat"]), bbox_wgs84)
        return False

    def _show_facility(lon: float, lat: float) -> bool:
        return bbox_wgs84 is None or _in_bbox_wgs84(lon, lat, bbox_wgs84)

    for pid, m in matches.items():
        if not _show_cams(m):
            continue
        c = m["cams"]
        ok = m.get("matched") == "yes"
        color = "green" if ok else "red"
        dist = m.get("scoring_value")
        dist_s = f"{float(dist):.2f} km" if dist is not None else "—"
        if m.get("match_source") == "corine":
            layer = "corine"
        elif m.get("match_source") == "riurbans":
            layer = "riurbans"
        elif m.get("osm_facility_id"):
            layer = "osm"
        elif m.get("match_source") == "uwwtd":
            layer = "uwwtd"
        elif m.get("eprtr_point_id"):
            layer = "eprtr"
        else:
            layer = m.get("match_layer", m.get("match_source") or "jrc")
        links = _facility_links(m)
        fac = links[0].get("facility_id") if links else (
            m.get("corine_facility_id")
            or m.get("uwwtd_facility_id")
            or m.get("eprtr_point_id")
            or m.get("jrc_point_id")
            or m.get("osm_facility_id")
            or "—"
        )
        n_links = len(links)
        popup_extra = f"<br>facility links: {n_links}" if n_links > 1 else ""
        folium.CircleMarker(
            location=[c["latitude"], c["longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                f"<b>CAMS {pid}</b><br>matched: {m.get('matched')}<br>"
                f"layer: {layer}<br>facility: {fac}{popup_extra}<br>distance: {dist_s}",
                max_width=280,
            ),
        ).add_to(fmap)
        if ok and links:
            src = str(m.get("match_source") or "")
            line_color = {
                "riurbans": "#6a1b9a",
                "jrc": "#333333",
                "eprtr": "#0088cc",
                "osm": "#3949ab",
                "corine": "#5d4037",
                "uwwtd": "#00695c",
            }.get(src, "#666666")
            for lk in links:
                info = lk.get("facility_info") or {}
                if info.get("lat") is None or info.get("lon") is None:
                    continue
                folium.PolyLine(
                    locations=[[c["latitude"], c["longitude"]], [float(info["lat"]), float(info["lon"])]],
                    color=line_color,
                    weight=2,
                    opacity=0.8,
                ).add_to(fmap)
        elif ok and m.get("match_source") == "corine" and m.get("corine_facility_info"):
            ci = m["corine_facility_info"]
            folium.PolyLine(
                locations=[[c["latitude"], c["longitude"]], [float(ci["lat"]), float(ci["lon"])]],
                color="#5d4037",
                weight=2,
                opacity=0.8,
            ).add_to(fmap)
        elif ok and m.get("match_source") == "uwwtd" and m.get("uwwtd_facility_info"):
            w = m["uwwtd_facility_info"]
            folium.PolyLine(
                locations=[[c["latitude"], c["longitude"]], [float(w["lat"]), float(w["lon"])]],
                color="#00695c",
                weight=2,
                opacity=0.85,
            ).add_to(fmap)
        elif ok and m.get("eprtr_point_info"):
            j = m["eprtr_point_info"]
            folium.PolyLine(
                locations=[[c["latitude"], c["longitude"]], [j["lat"], j["lon"]]],
                color="#0088cc",
                weight=2,
                opacity=0.8,
            ).add_to(fmap)
        elif ok and m.get("jrc_point_info"):
            j = m["jrc_point_info"]
            folium.PolyLine(
                locations=[[c["latitude"], c["longitude"]], [j["lat"], j["lon"]]],
                color="#333333",
                weight=2,
                opacity=0.7,
            ).add_to(fmap)
        elif ok and m.get("osm_facility_info"):
            oi = m["osm_facility_info"]
            folium.PolyLine(
                locations=[[c["latitude"], c["longitude"]], [float(oi["lat"]), float(oi["lon"])]],
                color="#3949ab",
                weight=2,
                opacity=0.75,
            ).add_to(fmap)
        elif not ok:
            nearest = _facility_info_for_line(m)
            if nearest is not None:
                folium.PolyLine(
                    locations=[[c["latitude"], c["longitude"]], [float(nearest["lat"]), float(nearest["lon"])]],
                    color="#e65100",
                    weight=2,
                    opacity=0.65,
                    dash_array="6, 8",
                ).add_to(fmap)

    ri_points: dict[str, dict[str, Any]] = {}
    for m in matches.values():
        for lk in _facility_links(m):
            fid = str(lk.get("facility_id") or "")
            if not fid:
                continue
            src = str(m.get("match_source") or "")
            if src == "riurbans" or fid.startswith("ri_"):
                info = dict(lk.get("facility_info") or {})
                info.setdefault("lon", lk.get("facility_lon"))
                info.setdefault("lat", lk.get("facility_lat"))
                ri_points[fid] = info

    for rid, r in ri_points.items():
        if r.get("lon") is None or r.get("lat") is None:
            continue
        if not _show_facility(float(r["lon"]), float(r["lat"])):
            continue
        linked = rid in assigned_ri
        lat, lon = _jitter_latlon(rid, float(r["lat"]), float(r["lon"]))
        links = ri_to_cams.get(rid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        gnfr = str(r.get("gnfr") or "")
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="#6a1b9a" if linked else "#ce93d8",
            fill=True,
            fill_color="#6a1b9a" if linked else "#f3e5f5",
            fill_opacity=0.88,
            weight=2,
            popup=folium.Popup(
                f"<b>RI-URBANS {gnfr}</b><br>id: {rid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    for jid, j in jrc_points.items():
        if not _show_facility(float(j["lon"]), float(j["lat"])):
            continue
        linked = jid in assigned_jrc
        lat, lon = _jitter_latlon(jid, float(j["lat"]), float(j["lon"]))
        links = jrc_to_cams.get(jid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="blue" if linked else "purple",
            fill=True,
            fill_color="blue" if linked else "purple",
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                f"<b>JRC {j.get('name_g', j.get('facility_name', ''))}</b><br>id: {jid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    for eid, e in eprtr_points.items():
        if not _show_facility(float(e["lon"]), float(e["lat"])):
            continue
        linked = eid in assigned_eprtr
        lat, lon = _jitter_latlon(eid, float(e["lat"]), float(e["lon"]))
        links = eprtr_to_cams.get(eid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="cyan" if linked else "darkorange",
            fill=True,
            fill_color="cyan" if linked else "darkorange",
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                f"<b>EPRTR {e.get('facility_name', '')}</b><br>id: {eid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    for oid, o in osm_facility_points.items():
        if not _show_facility(float(o["lon"]), float(o["lat"])):
            continue
        linked = oid in assigned_osm
        lat, lon = _jitter_latlon(oid, float(o["lat"]), float(o["lon"]))
        links = osm_to_cams.get(oid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        label = str(o.get("name") or "aerodrome")
        icao = str(o.get("icao") or "").strip()
        title = f"{label} ({icao})" if icao else label
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="#1a237e" if linked else "#7e57c2",
            fill=True,
            fill_color="#283593" if linked else "#b39ddb",
            fill_opacity=0.88,
            weight=2,
            popup=folium.Popup(
                f"<b>OSM {title}</b><br>id: {oid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    for uid, w in uwwtd_facility_points.items():
        if not _show_facility(float(w["lon"]), float(w["lat"])):
            continue
        linked = uid in assigned_uwwtd
        lat, lon = _jitter_latlon(uid, float(w["lat"]), float(w["lon"]))
        links = uwwtd_to_cams.get(uid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="#004d40" if linked else "#80cbc4",
            fill=True,
            fill_color="#00695c" if linked else "#b2dfdb",
            fill_opacity=0.88,
            weight=2,
            popup=folium.Popup(
                f"<b>UWWTD {w.get('facility_name', '')}</b><br>id: {uid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    for cid, cf in corine_facility_points.items():
        if not _show_facility(float(cf["lon"]), float(cf["lat"])):
            continue
        linked = cid in assigned_corine
        lat, lon = _jitter_latlon(cid, float(cf["lat"]), float(cf["lon"]))
        links = corine_to_cams.get(cid, [])
        link_lines = "<br>".join(f"CAMS {p}: {st} ({d:.1f} km)" for p, st, d in links[:8])
        l3 = cf.get("l3", "")
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if linked else 8,
            color="#4e342e" if linked else "#a1887f",
            fill=True,
            fill_color="#5d4037" if linked else "#d7ccc8",
            fill_opacity=0.88,
            weight=2,
            popup=folium.Popup(
                f"<b>CORINE L{l3}</b><br>id: {cid}<br>"
                f"matched to CAMS: {'yes' if linked else 'no'}"
                + (f"<br><br>{link_lines}" if link_lines else ""),
                max_width=320,
            ),
        ).add_to(fmap)

    legend_rows = [
        "<b>CAMS</b> <span style='color:green'>&#9679;</span> matched "
        "<span style='color:red'>&#9679;</span> not matched",
        "<span style='color:#e65100'>- -</span> dashed = unmatched CAMS → nearest facility",
        "<b>RI-URBANS</b> <span style='color:#6a1b9a'>&#9679;</span> matched "
        "<span style='color:#ce93d8'>&#9679;</span> unmatched pool",
        "<b>JRC</b> <span style='color:blue'>&#9679;</span> matched "
        "<span style='color:purple'>&#9679;</span> unmatched pool",
        "<b>EPRTR</b> <span style='color:cyan'>&#9679;</span> matched "
        "<span style='color:darkorange'>&#9679;</span> unmatched pool",
    ]
    if uwwtd_facility_points:
        legend_rows.append(
            "<b>UWWTD</b> <span style='color:#00695c'>&#9679;</span> matched "
            "<span style='color:#b2dfdb'>&#9679;</span> unmatched pool"
        )
    if osm_facility_points:
        legend_rows.append(
            "<b>OSM aerodrome</b> <span style='color:#283593'>&#9679;</span> matched "
            "<span style='color:#b39ddb'>&#9679;</span> unmatched pool"
        )
    if osm_polygons_gdf is not None and not osm_polygons_gdf.empty:
        legend_rows.append("<b>OSM polygons</b> light blue fill = all GPKG polygon rows in map extent")
    if corine_facility_points:
        legend_rows.append(
            "<b>CORINE airport</b> <span style='color:#5d4037'>&#9679;</span> matched "
            "<span style='color:#d7ccc8'>&#9679;</span> unmatched pool"
        )
    legend_html = (
        "<div style='position: fixed; bottom: 24px; left: 12px; z-index: 9999; "
        "background: white; padding: 10px 12px; border: 1px solid #888; font-size: 12px;'>"
        + "<br>".join(legend_rows)
        + "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    return out_html


def write_cams_jrc_match_map(
    matches: dict[int, dict[str, Any]],
    jrc_points: dict[str, dict[str, Any]],
    out_html: Path,
    *,
    eprtr_points: dict[str, dict[str, Any]] | None = None,
    osm_facility_points: dict[str, dict[str, Any]] | None = None,
    osm_polygons_gdf: gpd.GeoDataFrame | None = None,
    corine_facility_points: dict[str, dict[str, Any]] | None = None,
    uwwtd_facility_points: dict[str, dict[str, Any]] | None = None,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> Path:
    return write_point_match_map(
        matches,
        jrc_points,
        out_html,
        eprtr_points=eprtr_points,
        osm_facility_points=osm_facility_points,
        osm_polygons_gdf=osm_polygons_gdf,
        corine_facility_points=corine_facility_points,
        uwwtd_facility_points=uwwtd_facility_points,
        bbox_wgs84=bbox_wgs84,
    )
