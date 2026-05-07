"""Build OSM-based facility candidates for GNFR H (aviation) CAMS point matching.

Candidate polygons come from ``paths.yaml`` → ``osm.aviation`` (``*_layers.gpkg``).
Optional apron/terminal layers and aerodrome nodes are configured under
``sector_cfg["point_matching"]`` — see ``README.md`` in this package.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_osm_tags(tags_raw: object) -> dict[str, str]:
    if tags_raw is None or (isinstance(tags_raw, float) and math.isnan(tags_raw)):
        return {}
    if isinstance(tags_raw, dict):
        return {str(k): str(v) for k, v in tags_raw.items()}
    s = str(tags_raw).strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tag_nonempty(tags: dict[str, str], key: str) -> bool:
    v = tags.get(key)
    return v is not None and str(v).strip() != ""


def _norm_name(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    return str(name).strip().lower()


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _geom_match_point_m3035(
    aerodrome_poly: Any,
    apron_union_clipped: Any | None,
) -> Any:
    """Return a Shapely point in EPSG:3035 for ``match_location`` logic."""
    if apron_union_clipped is not None and not apron_union_clipped.is_empty:
        u = apron_union_clipped
        c = u.centroid
        if u.contains(c):
            return c
        return u.representative_point()
    c0 = aerodrome_poly.centroid
    if aerodrome_poly.contains(c0):
        return c0
    return aerodrome_poly.representative_point()


def _collect_apron_terminal_union(
    aerodrome_poly: Any,
    aux_gpkg: Path,
    layer_names: list[str],
    crs_target: Any,
) -> Any | None:
    if not layer_names or not aux_gpkg.is_file():
        return None
    parts: list[Any] = []
    for lyr in layer_names:
        try:
            aux = gpd.read_file(aux_gpkg, layer=str(lyr))
        except Exception:
            continue
        if aux.empty or aux.geometry.is_empty.all():
            continue
        aux = aux.to_crs(crs_target)
        hits = aux[aux.geometry.intersects(aerodrome_poly)]
        if hits.empty:
            continue
        try:
            u = unary_union(hits.geometry.tolist())
        except Exception:
            continue
        inter = u.intersection(aerodrome_poly)
        if inter.is_empty:
            continue
        parts.append(inter)
    if not parts:
        return None
    try:
        return unary_union(parts)
    except Exception:
        return None


def build_aviation_facility_candidates(
    *,
    repo_root: Path,
    paths_resolved: dict[str, Any],
    sector_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Return ``(facilities_df, resolved_source_gpkg)`` for :func:`load_facility_candidates_for_sector`.

    Columns align with :func:`PROXY.core.matching.run_matching` expectations, plus diagnostics
    ``icao``, ``iata``, ``osm_source``, ``area_km2``, ``polygon_centroid_lon``, ``polygon_centroid_lat``,
    ``osm_element_type``, ``osm_numeric_id``.
    """
    pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    osm_block = paths_resolved.get("osm") or {}
    rel = pm.get("aviation_gpkg") or osm_block.get("aviation")
    if not rel:
        raise ValueError(
            "point_matching.facility_pool=aviation requires paths.yaml ``osm.aviation`` "
            "or sector ``point_matching.aviation_gpkg``."
        )
    gpkg = Path(str(rel)) if Path(str(rel)).is_absolute() else repo_root / str(rel)
    if not gpkg.is_file():
        raise FileNotFoundError(f"Aviation OSM GPKG not found: {gpkg}")

    poly_layer = str(pm.get("aviation_polygon_layer", "osm_aviation_airport_polygons"))
    try:
        gdf = gpd.read_file(gpkg, layer=poly_layer)
    except Exception as exc:
        layers = [x[0] for x in gpd.list_layers(gpkg)]
        raise ValueError(
            f"Cannot read layer {poly_layer!r} from {gpkg} (available: {layers})."
        ) from exc

    if gdf.empty:
        return (
            pd.DataFrame(
                columns=[
                    "facility_id",
                    "facility_name",
                    "eprtr_sector_code",
                    "pollutant",
                    "longitude",
                    "latitude",
                    "reporting_year",
                    "_registry",
                ]
            ),
            str(gpkg.resolve()),
        )

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3035")

    fam_col = "aviation_family" if "aviation_family" in gdf.columns else None
    keep_fam = {"aerodrome", "landuse_aerodrome"}
    if fam_col:
        gdf = gdf[gdf[fam_col].astype(str).str.lower().isin(keep_fam)].copy()

    aux_gpkg = Path(str(pm.get("aviation_terminal_apron_gpkg", gpkg)))
    if not aux_gpkg.is_file():
        aux_gpkg = gpkg
    apron_layers = list(pm.get("aviation_terminal_apron_layers") or [])

    rows: list[dict[str, Any]] = []
    poly_meta_for_dedupe: list[dict[str, Any]] = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        tags = _parse_osm_tags(row.get("osm_tags"))
        if fam_col and str(row.get(fam_col, "")).strip().lower() == "military_airfield":
            continue
        if _tag_nonempty(tags, "military"):
            continue
        area_km2 = float(geom.area) / 1e6
        if area_km2 < 0.5:
            continue

        apron_u = _collect_apron_terminal_union(geom, aux_gpkg, apron_layers, gdf.crs)
        match_pt_m = _geom_match_point_m3035(geom, apron_u)

        cen = geom.centroid
        gdiag = gpd.GeoDataFrame(
            {"geometry": [match_pt_m, cen]},
            crs=gdf.crs,
        ).to_crs("EPSG:4326")
        match_ll = gdiag.geometry.iloc[0]
        cen_ll = gdiag.geometry.iloc[1]

        icao = (row.get("icao") or tags.get("icao") or "").strip().upper() or None
        iata = (row.get("iata") or tags.get("iata") or "").strip().upper() or None
        name = str(row.get("name") or tags.get("name") or "").strip() or "aerodrome"
        etype = str(row.get("osm_element_type") or "way")
        eid = row.get("osm_element_id")
        if pd.isna(eid) or eid is None:
            eid = int(idx)
        eid_i = int(eid)
        fac_id = f"osm:{etype}:{eid_i}"

        rows.append(
            {
                "facility_id": fac_id,
                "facility_name": name,
                "eprtr_sector_code": 0,
                "pollutant": "CO2",
                "longitude": float(match_ll.x),
                "latitude": float(match_ll.y),
                "reporting_year": int(pm.get("reference_year", 2019)),
                "_registry": "OSM_AVIATION_POLY",
                "icao": icao or "",
                "iata": iata or "",
                "osm_source": "polygon",
                "area_km2": area_km2,
                "polygon_centroid_lon": float(cen_ll.x),
                "polygon_centroid_lat": float(cen_ll.y),
                "osm_element_type": etype,
                "osm_numeric_id": eid_i,
            }
        )
        poly_meta_for_dedupe.append(
            {
                "icao": icao,
                "name_norm": _norm_name(name),
                "lon": float(match_ll.x),
                "lat": float(match_ll.y),
            }
        )

    poly_icoes = {m["icao"] for m in poly_meta_for_dedupe if m["icao"]}

    nodes_gpkg = pm.get("aviation_aerodrome_nodes_gpkg")
    if nodes_gpkg:
        ng_path = Path(str(nodes_gpkg)) if Path(str(nodes_gpkg)).is_absolute() else repo_root / str(nodes_gpkg)
        node_layer = str(pm.get("aviation_aerodrome_nodes_layer", "osm_aviation_aerodrome_nodes"))
        if ng_path.is_file():
            try:
                ndf = gpd.read_file(ng_path, layer=node_layer)
            except Exception:
                ndf = gpd.read_file(ng_path)
            if not ndf.empty:
                if ndf.crs is None:
                    ndf = ndf.set_crs("EPSG:4326")
                ndf = ndf.to_crs("EPSG:3035")
                node_buf_m = float(pm.get("aviation_node_buffer_m", 1000.0))
                for _, nrow in ndf.iterrows():
                    g0 = nrow.geometry
                    if g0 is None or g0.is_empty:
                        continue
                    tags = _parse_osm_tags(nrow.get("osm_tags"))
                    if str(tags.get("aeroway", "")).strip().lower() != "aerodrome":
                        continue
                    if _tag_nonempty(tags, "military"):
                        continue
                    icao_n = (tags.get("icao") or nrow.get("icao") or "").strip().upper() or None
                    name_n = str(nrow.get("name") or tags.get("name") or "").strip() or "aerodrome"
                    nn = _norm_name(name_n)
                    if icao_n and icao_n in poly_icoes:
                        continue
                    pt_ll = gpd.GeoDataFrame(geometry=[g0], crs=ndf.crs).to_crs(
                        "EPSG:4326"
                    ).geometry.iloc[0]
                    lon_n = float(pt_ll.x)
                    lat_n = float(pt_ll.y)
                    skip = False
                    for m in poly_meta_for_dedupe:
                        if nn and m["name_norm"] and nn == m["name_norm"]:
                            if _haversine_km(lon_n, lat_n, m["lon"], m["lat"]) < 2.0:
                                skip = True
                                break
                    if skip:
                        continue
                    etype = str(nrow.get("osm_element_type") or "node")
                    eid = nrow.get("osm_element_id")
                    if pd.isna(eid) or eid is None:
                        continue
                    eid_i = int(eid)
                    buf_area_km2 = math.pi * (node_buf_m / 1000.0) ** 2
                    rows.append(
                        {
                            "facility_id": f"osm:{etype}:{eid_i}",
                            "facility_name": name_n,
                            "eprtr_sector_code": 0,
                            "pollutant": "CO2",
                            "longitude": float(lon_n),
                            "latitude": float(lat_n),
                            "reporting_year": int(pm.get("reference_year", 2019)),
                            "_registry": "OSM_AVIATION_NODE",
                            "icao": icao_n or "",
                            "iata": (tags.get("iata") or nrow.get("iata") or "").strip().upper() or "",
                            "osm_source": "node_buffer",
                            "area_km2": float(buf_area_km2),
                            "polygon_centroid_lon": float(lon_n),
                            "polygon_centroid_lat": float(lat_n),
                            "osm_element_type": etype,
                            "osm_numeric_id": eid_i,
                        }
                    )
        else:
            warnings.warn(f"aviation_aerodrome_nodes_gpkg not found: {ng_path}", stacklevel=1)

    if not rows:
        return (
            pd.DataFrame(
                columns=[
                    "facility_id",
                    "facility_name",
                    "eprtr_sector_code",
                    "pollutant",
                    "longitude",
                    "latitude",
                    "reporting_year",
                    "_registry",
                ]
            ),
            str(gpkg.resolve()),
        )

    out = pd.DataFrame(rows)
    return out, str(gpkg.resolve())


def write_aviation_match_diagnostic_png(
    *,
    matches_csv: Path,
    out_png: Path,
    title: str = "Aviation CAMS ↔ OSM matches",
) -> Path | None:
    """Optional matplotlib map: CAMS points, match locations, and connecting segments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not matches_csv.is_file():
        return None
    df = pd.read_csv(matches_csv)
    need = {"cams_longitude", "cams_latitude", "facility_longitude", "facility_latitude"}
    if not need.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(8, 9), dpi=120)
    lon0 = df["cams_longitude"].astype(float)
    lat0 = df["cams_latitude"].astype(float)
    lon1 = df["facility_longitude"].astype(float)
    lat1 = df["facility_latitude"].astype(float)
    for i in range(len(df)):
        ax.plot([lon0.iloc[i], lon1.iloc[i]], [lat0.iloc[i], lat1.iloc[i]], color="#90caf9", lw=0.8, zorder=1)
    ax.scatter(lon0, lat0, c="#c62828", s=14, zorder=3, label="CAMS")
    ax.scatter(lon1, lat1, c="#1565c0", s=14, zorder=3, label="Match location")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png
