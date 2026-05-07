"""
Match CAMS GNFR D (fugitive) point sources to E-PRTR facilities using the same scoring
mechanism as ``Waste/Auxiliaries/waste_cams_proxy_match.py``.

Candidate pool (fuel-exploitation scope, same Annex allowlist as ``cams_D_greece_map``):
  - **lcp_energy** — F5_2 LCP plants (higher class score; preferred when in range).
  - **fugitive_air_release** — F1_4 AIR facilities with matching Annex I codes (fallback).

Rasters: relocates matched emissions (kg/yr) to facility coordinates; unmatched keep CAMS lon/lat.

Use ``--write-map`` to save an interactive Folium HTML (CAMS points, E-PRTR facilities, polylines).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Tunables — mirror Waste logic: class base score + distance (0–40) + evidence bonus cap 25
FUGITIVE_ELIGIBLE_CATEGORIES: tuple[str, ...] = ("lcp_energy", "fugitive_air_release")

CLASS_SCORES: dict[str, float] = {
    "lcp_energy": 100.0,
    "fugitive_air_release": 70.0,
}

MAX_DISTANCE_KM: dict[str, float] = {
    "lcp_energy": 10.0,
    "fugitive_air_release": 7.0,
}

ACCEPT_MIN_BEST_SCORE = 80.0
ACCEPT_MIN_MARGIN_VS_SECOND = 15.0
EVIDENCE_BONUS_CAP = 25.0


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_waste_proxy_match():
    p = _root() / "Waste" / "Auxiliaries" / "waste_cams_proxy_match.py"
    spec = importlib.util.spec_from_file_location("waste_cams_proxy_match", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def _load_cams_d_map():
    p = Path(__file__).resolve().parent / "cams_D_greece_map.py"
    spec = importlib.util.spec_from_file_location("cams_d_greece_map", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


_wm = _load_waste_proxy_match()


def distance_score_component(distance_km: float, class_threshold_km: float) -> float:
    return _wm.distance_score_component(distance_km, class_threshold_km)


def fugitive_evidence_bonus(row: pd.Series) -> float:
    bonus = 0.0
    f5 = bool(row.get("from_f5_2"))
    f1 = bool(row.get("from_f1_4"))
    n_src = int(f5) + int(f1)
    if n_src >= 2:
        bonus += 15.0
    elif n_src == 1:
        bonus += 5.0
    if bool(row.get("has_air_release")) and str(row.get("primary_category")) == "fugitive_air_release":
        bonus += 5.0
    return float(min(bonus, EVIDENCE_BONUS_CAP))


def score_candidate(
    distance_km: float,
    facility_row: pd.Series,
    *,
    distance_scale: float = 1.0,
) -> tuple[float, float, float, float]:
    cat = str(facility_row.get("primary_category", ""))
    base = CLASS_SCORES.get(cat, 0.0)
    raw_thr = MAX_DISTANCE_KM.get(cat, 0.0)
    thr = float(raw_thr) * float(distance_scale) if raw_thr else 0.0
    dscore = distance_score_component(distance_km, thr)
    ev = fugitive_evidence_bonus(facility_row)
    total = base + dscore + ev
    return total, base, dscore, ev


def generate_candidate_pairs(
    cams: pd.DataFrame,
    facilities: pd.DataFrame,
    *,
    distance_scale: float = 1.0,
) -> pd.DataFrame:
    if cams.empty or facilities.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    fac_lat = facilities["latitude"].astype(np.float64).values
    fac_lon = facilities["longitude"].astype(np.float64).values

    for _, cr in cams.iterrows():
        sid = str(cr.get("cams_source_id", ""))
        if not sid:
            continue
        clat = float(cr["cams_lat"])
        clon = float(cr["cams_lon"])
        dist_all = _wm.haversine_km(clat, clon, fac_lat, fac_lon)

        for fi, (_, fr) in enumerate(facilities.iterrows()):
            cat = str(fr.get("primary_category", ""))
            d_km = float(dist_all[fi])
            raw_thr = MAX_DISTANCE_KM.get(cat)
            thr = float(raw_thr) * float(distance_scale) if raw_thr is not None else None
            if thr is None or d_km > thr:
                continue
            total, cscore, dscore, ev = score_candidate(d_km, fr, distance_scale=distance_scale)
            mid = str(fr.get("master_id", ""))
            rows.append(
                {
                    "cams_source_id": sid,
                    "facility_id": mid,
                    "facility_category": cat,
                    "distance_km": d_km,
                    "class_score": cscore,
                    "distance_score": dscore,
                    "evidence_bonus": ev,
                    "total_score": total,
                    "facility_row_index": fi,
                }
            )

    return pd.DataFrame(rows)


def resolve_lcp_tie_nearest(
    g: pd.DataFrame,
    *,
    best_score: float,
    second_score: float,
) -> pd.Series | None:
    """
    When total_score ties between two+ ``lcp_energy`` candidates (typical with identical
    base+distance+evidence), pick the nearest installation. Same role as UWWTD tie-break in Waste.
    """
    if g.empty or len(g) < 2:
        return None
    if abs(best_score - second_score) > 1e-4:
        return None
    g0, g1 = g.iloc[0], g.iloc[1]
    if str(g0["facility_category"]) != "lcp_energy":
        return None
    if str(g1["facility_category"]) != "lcp_energy":
        return None
    if abs(float(g0["total_score"]) - float(g1["total_score"])) > 1e-4:
        return None
    mx = float(g0["total_score"])
    pool = g[np.isclose(g["total_score"].astype(float), mx, rtol=0, atol=1e-4)].copy()
    if len(pool) < 2:
        return None
    if not (pool["facility_category"].astype(str) == "lcp_energy").all():
        return None
    return pool.sort_values(["distance_km", "facility_id"], ascending=[True, True]).iloc[0]


def resolve_lcp_top_two_nearest(g: pd.DataFrame) -> pd.Series | None:
    """
    If the two strongest candidates are both ``lcp_energy`` but scores differ slightly
    (ambiguous margin), pick the nearer plant — matches inspection that GNFR D points
    should anchor to the closest LCP when multiple are plausible.
    """
    if len(g) < 2:
        return None
    g0, g1 = g.iloc[0], g.iloc[1]
    if str(g0["facility_category"]) != "lcp_energy" or str(g1["facility_category"]) != "lcp_energy":
        return None
    if float(g0["total_score"]) < ACCEPT_MIN_BEST_SCORE:
        return None
    pair = g.iloc[0:2].copy()
    return pair.sort_values(["distance_km", "facility_id"], ascending=[True, True]).iloc[0]


def assign_best_match_fugitive(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like ``waste_cams_proxy_match.assign_best_match`` with LCP tie-break before UWWTD rule."""
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    cand = candidates.copy()
    cand = cand.sort_values(
        ["cams_source_id", "total_score", "distance_km"],
        ascending=[True, False, True],
    )
    cand["candidate_rank"] = cand.groupby("cams_source_id")["total_score"].rank(
        method="first", ascending=False
    ).astype(int)

    match_rows: list[dict[str, Any]] = []

    for sid, g in cand.groupby("cams_source_id", sort=False):
        g = g.sort_values("candidate_rank", ascending=True)
        if g.empty:
            continue
        g0 = g.iloc[0]
        best_score = float(g0["total_score"])
        second_score = float(g.iloc[1]["total_score"]) if len(g) > 1 else float("-inf")
        margin = best_score - second_score if len(g) > 1 else float("inf")

        chosen = g0
        matched = False
        status = "unmatched_below_score_threshold"

        if best_score < ACCEPT_MIN_BEST_SCORE:
            status = "unmatched_below_score_threshold"
            matched = False
        elif len(g) > 1 and margin < ACCEPT_MIN_MARGIN_VS_SECOND:
            pick = resolve_lcp_tie_nearest(g, best_score=best_score, second_score=second_score)
            if pick is None:
                pick = resolve_lcp_top_two_nearest(g)
            if pick is not None:
                chosen = pick
                best_score = float(chosen["total_score"])
                others = g[g["facility_id"].astype(str) != str(chosen["facility_id"])]
                second_score = float(others["total_score"].max()) if len(others) else float("-inf")
                margin = best_score - second_score if len(others) else float("inf")
                status = "matched"
                matched = True
            else:
                pick = _wm.resolve_uwwtd_ambiguous_nearest(
                    g, best_score=best_score, second_score=second_score
                )
                if pick is not None:
                    chosen = pick
                    best_score = float(chosen["total_score"])
                    others = g[g["facility_id"].astype(str) != str(chosen["facility_id"])]
                    second_score = float(others["total_score"].max()) if len(others) else float("-inf")
                    margin = best_score - second_score if len(others) else float("inf")
                    status = "matched"
                    matched = True
                else:
                    status = "ambiguous_multiple_candidates"
                    matched = False
        else:
            status = "matched"
            matched = True

        match_rows.append(
            {
                "cams_source_id": sid,
                "match_score": best_score,
                "second_best_score": second_score if len(g) > 1 else float("nan"),
                "score_margin_vs_second": margin if len(g) > 1 else float("nan"),
                "match_status": status,
                "matched_master_id": str(chosen["facility_id"]) if matched else "",
                "matched_category": str(chosen["facility_category"]) if matched else "",
                "distance_km": float(chosen["distance_km"]) if matched else float("nan"),
                "best_master_id": str(g0["facility_id"]),
                "best_category": str(g0["facility_category"]),
                "best_distance_km": float(g0["distance_km"]),
                "candidate_rank1_class_score": float(g0["class_score"]),
                "candidate_rank1_distance_score": float(g0["distance_score"]),
                "candidate_rank1_evidence_bonus": float(g0["evidence_bonus"]),
            }
        )

    match_df = pd.DataFrame(match_rows)
    return match_df, cand


def filter_eligible_fugitive_facilities(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty or "primary_category" not in master.columns:
        return pd.DataFrame()
    m = master["primary_category"].astype(str).isin(FUGITIVE_ELIGIBLE_CATEGORIES)
    return master.loc[m].copy()


def build_fugitive_facility_master(
    *,
    country_arg: str,
    all_countries: bool,
    bbox: tuple[float, float, float, float] | None,
    f5_csv: Path,
    f1_csv: Path,
    eprtr_pollutant: str,
    cd: Any,
) -> pd.DataFrame:
    f5 = cd.load_f5_lcp_plants(
        f5_csv,
        all_countries=all_countries,
        country_arg=country_arg,
        bbox=bbox,
    )
    rows: list[dict[str, Any]] = []
    if not f5.empty:
        for _, r in f5.iterrows():
            rows.append(
                {
                    "FacilityInspireId": str(r["LCPInspireId"]),
                    "facility_name": str(r.get("facilityName", "") or ""),
                    "longitude": float(r["Longitude"]),
                    "latitude": float(r["Latitude"]),
                    "primary_category": "lcp_energy",
                    "from_f5_2": True,
                    "from_f1_4": False,
                    "from_f4_2": False,
                    "from_f6_1": False,
                    "from_f7_1": False,
                    "has_air_release": False,
                    "is_incinerator": False,
                    "is_co_incinerator": False,
                    "is_wastewater": False,
                    "is_landfill": False,
                    "is_wte": False,
                }
            )

    f1 = cd.load_eprtr_facilities_aggregated(
        f1_csv,
        all_countries=all_countries,
        country_arg=country_arg,
        pollutant=eprtr_pollutant,
        sector_code=None,
        bbox=bbox,
        annex_allowlist=cd.EPRTR_FUEL_EXPLOITATION_ANNEX_CODES,
    )
    if not f1.empty:
        for _, r in f1.iterrows():
            rows.append(
                {
                    "FacilityInspireId": str(r["FacilityInspireId"]),
                    "facility_name": str(r.get("facilityName", "") or ""),
                    "longitude": float(r["Longitude"]),
                    "latitude": float(r["Latitude"]),
                    "primary_category": "fugitive_air_release",
                    "from_f5_2": False,
                    "from_f1_4": True,
                    "from_f4_2": False,
                    "from_f6_1": False,
                    "from_f7_1": False,
                    "has_air_release": True,
                    "is_incinerator": False,
                    "is_co_incinerator": False,
                    "is_wastewater": False,
                    "is_landfill": False,
                    "is_wte": False,
                }
            )

    master = pd.DataFrame(rows)
    if master.empty:
        return master
    master.insert(0, "master_id", [f"M{i+1:06d}" for i in range(len(master))])
    return master


def load_cams_d_points(
    nc_path: Path,
    *,
    country_iso3: str | None,
    all_countries: bool,
    bbox: tuple[float, float, float, float] | None,
    pollutant: str | None,
) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    try:
        emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
        st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
        ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
        lon = np.asarray(ds["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(ds["latitude_source"].values).ravel().astype(np.float64)
        idx_d = 4
        m = (emis == idx_d) & (st == 2)
        if not all_countries and country_iso3:
            raw = ds["country_id"].values
            codes: list[str] = []
            for x in raw:
                if isinstance(x, bytes):
                    codes.append(x.decode("utf-8", "replace").strip())
                else:
                    codes.append(str(x).strip())
            c1 = codes.index(country_iso3.strip().upper()) + 1
            m = m & (ci == c1)
        if bbox is not None:
            lon0, lat0, lon1, lat1 = bbox
            m = m & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
        idx = np.flatnonzero(m)

        pol_var: str | None = None
        if pollutant:
            want = str(pollutant).strip().lower()
            pmap = {str(v).lower(): str(v) for v in ds.data_vars}
            if want not in pmap:
                raise SystemExit(
                    f"CAMS NetCDF has no pollutant variable matching {pollutant!r}. "
                    f"Examples: {', '.join(list(pmap.keys())[:16])}"
                )
            pol_var = pmap[want]
        pol_arr = (
            np.asarray(ds[pol_var].values).ravel().astype(np.float64) if pol_var else None
        )

        rows: list[dict[str, Any]] = []
        for i in idx:
            ii = int(i)
            row: dict[str, Any] = {
                "cams_lon": float(lon[ii]),
                "cams_lat": float(lat[ii]),
                "nc_source_index": ii,
                "cams_source_id": f"CAMS_D_{ii:07d}",
            }
            if pol_arr is not None:
                row["cams_pollutant_kg"] = (
                    float(pol_arr[ii]) if ii < pol_arr.size else float("nan")
                )
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        ds.close()


def run_fugitive_proxy_workflow(
    master: pd.DataFrame,
    cams: pd.DataFrame,
    *,
    out_dir: Path,
    pixel_size_deg: float,
    tif_filename: str = "Fugitive_pointsource.tif",
    distance_scale: float = 1.0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dscale = float(distance_scale)
    if dscale <= 0:
        raise ValueError("distance_scale must be positive.")

    elig = filter_eligible_fugitive_facilities(master)
    fac_pool = elig.copy()

    print(
        f"Fugitive CAMS / facility: {len(cams)} CAMS GNFR D points, "
        f"{len(elig)} facilities (LCP + F1_4 annex filter)."
    )
    if dscale != 1.0:
        print(f"  Distance scale: {dscale:g} (max km per class = table x {dscale:g})")

    candidates = generate_candidate_pairs(cams, fac_pool, distance_scale=dscale)
    print(f"  Candidate pairs (within class distance thresholds): {len(candidates)}")

    assign_partial, cand_ranked = assign_best_match_fugitive(candidates)

    known_ids = set(cams["cams_source_id"].astype(str))
    assigned_ids = (
        set(assign_partial["cams_source_id"].astype(str)) if not assign_partial.empty else set()
    )
    missing = known_ids - assigned_ids
    extra = [
        {
            "cams_source_id": sid,
            "match_score": float("nan"),
            "second_best_score": float("nan"),
            "score_margin_vs_second": float("nan"),
            "match_status": "unmatched_no_candidate",
            "matched_master_id": "",
            "matched_category": "",
            "distance_km": float("nan"),
            "best_master_id": "",
            "best_category": "",
            "best_distance_km": float("nan"),
            "candidate_rank1_class_score": float("nan"),
            "candidate_rank1_distance_score": float("nan"),
            "candidate_rank1_evidence_bonus": float("nan"),
        }
        for sid in missing
    ]
    if extra:
        assign_partial = pd.concat([assign_partial, pd.DataFrame(extra)], ignore_index=True)

    assign_partial = _wm.resolve_unique_facility_per_match(assign_partial, cand_ranked)
    n_excl = int(
        (assign_partial["match_status"].astype(str) == "unmatched_facility_exclusive_loss").sum()
    )
    if n_excl:
        print(
            f"  Facility-exclusive rule: {n_excl} CAMS point(s) unmatched "
            "(lost duplicate facility claim)."
        )

    match_table, diag = _wm.build_match_tables(cams, fac_pool, candidates, assign_partial)

    vc = match_table["match_status"].value_counts()
    print("  Match status counts:")
    for k, v in vc.items():
        print(f"    {k}: {int(v)}")
    n_cams = len(match_table)
    n_matched = int((match_table["match_status"].astype(str) == "matched").sum())
    if n_cams:
        print(
            f"  Matched CAMS points: {n_matched} / {n_cams} ({100.0 * n_matched / n_cams:.1f}%)."
        )

    match_path = out_dir / "fugitive_cams_facility_proxy_matches.csv"
    cand_path = out_dir / "fugitive_cams_facility_proxy_candidates.csv"
    match_table.to_csv(match_path, index=False)
    cand_ranked.to_csv(cand_path, index=False)
    print(f"  Wrote {match_path.name} ({len(match_table)} rows)")
    print(f"  Wrote {cand_path.name} ({len(cand_ranked)} rows)")

    proxy_df = _wm.build_relocated_point_proxy(match_table)
    tif_path = out_dir / tif_filename
    _wm.rasterize_proxy_points_to_tif(proxy_df, tif_path, pixel_size_deg=pixel_size_deg)
    print(
        f"  Wrote GeoTIFF {tif_path} (EPSG:4326, pixel {pixel_size_deg} deg; band = summed kg/yr)."
    )

    fcounts = _wm.facility_cams_match_counts(match_table)

    preferred = [
        "cams_source_id",
        "cams_lon",
        "cams_lat",
        "matched_lon",
        "matched_lat",
        "matched_facility_id",
        "matched_master_id",
        "matched_facility_name",
        "matched_category",
        "distance_km",
        "match_score",
        "match_status",
        "used_original_cams_coordinate",
        "cams_pollutant_kg",
    ]
    cols = [c for c in preferred if c in match_table.columns] + [
        c for c in match_table.columns if c not in preferred
    ]
    match_table = match_table[cols]

    return {
        "match_table": match_table,
        "candidate_table": cand_ranked,
        "proxy_df": proxy_df,
        "eligible_facilities": elig,
        "facility_candidate_pool": fac_pool,
        "facility_match_counts": fcounts,
        "paths": {"matches_csv": match_path, "candidates_csv": cand_path, "tif": tif_path},
    }


def build_fugitive_proxy_link_map_html(
    match_table: pd.DataFrame,
    master: pd.DataFrame,
    out_html: Path,
    *,
    map_zoom: int = 7,
) -> None:
    """
    Folium map: CAMS GNFR D points, E-PRTR candidate facilities (LCP vs F1), and
    polylines CAMS -> matched facility (same styling helpers as Waste proxy map).
    """
    import folium

    if match_table.empty and master.empty:
        raise ValueError("match_table and master both empty.")

    lats: list[float] = []
    lons: list[float] = []
    if not master.empty:
        lats.extend(master["latitude"].astype(float).tolist())
        lons.extend(master["longitude"].astype(float).tolist())
    if not match_table.empty and "cams_lat" in match_table.columns:
        lats.extend(match_table["cams_lat"].astype(float).tolist())
        lons.extend(match_table["cams_lon"].astype(float).tolist())
    center_lat = float(np.mean(lats)) if lats else 39.0
    center_lon = float(np.mean(lons)) if lons else 22.0

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(map_zoom),
        tiles="OpenStreetMap",
    )
    folium.TileLayer("CartoDB positron", name="Map (light)", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satellite (Esri)",
        overlay=False,
        control=True,
    ).add_to(fmap)

    fg_cams = folium.FeatureGroup(name="CAMS GNFR D (points)", show=True)
    for _, r in match_table.iterrows():
        try:
            clat, clon = float(r["cams_lat"]), float(r["cams_lon"])
        except (TypeError, ValueError, KeyError):
            continue
        st = str(r.get("match_status", ""))
        if st == "matched":
            color, fill = "#238b45", "#a1d99b"
        elif st == "ambiguous_multiple_candidates":
            color, fill = "#d73027", "#fc8d59"
        else:
            color, fill = "#756bb1", "#bcbddc"
        kg = r.get("cams_pollutant_kg", "")
        kg_s = _fmt_popup_kg(kg)
        pop = (
            "<div style='font-size:13px;min-width:220px'>"
            "<b>CAMS GNFR D</b> point<br/>"
            f"<b>id</b> {html.escape(str(r.get('cams_source_id','')))}<br/>"
            f"<b>pollutant (kg/yr)</b> {kg_s}<br/>"
            f"<b>match_status</b> {html.escape(st)}<br/>"
            f"<b>score</b> {html.escape(str(r.get('match_score','')))}"
            "</div>"
        )
        folium.CircleMarker(
            location=[clat, clon],
            radius=9,
            color=color,
            weight=2,
            fill=True,
            fill_color=fill,
            fill_opacity=0.9,
            popup=folium.Popup(pop, max_width=320),
        ).add_to(fg_cams)
    fg_cams.add_to(fmap)

    fg_fac = folium.FeatureGroup(name="E-PRTR (LCP + F1 annex)", show=True)
    for _, r in master.iterrows():
        try:
            la, lo = float(r["latitude"]), float(r["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        cat = str(r.get("primary_category", ""))
        if cat == "lcp_energy":
            color, fill = "#08519c", "#6baed6"
            label = "F5_2 LCP"
        else:
            color, fill = "#b35806", "#fdae61"
            label = "F1_4 AIR"
        nm = html.escape(str(r.get("facility_name", "") or "")[:80])
        mid = html.escape(str(r.get("master_id", "")))
        fid = html.escape(str(r.get("FacilityInspireId", "") or ""))
        pop = (
            f"<div style='font-size:13px;min-width:200px'><b>{label}</b><br/>"
            f"<b>{nm}</b><br/>"
            f"<b>master_id</b> {mid}<br/>"
            f"<b>id</b><br/><span style='word-break:break-all'>{fid}</span></div>"
        )
        folium.CircleMarker(
            location=[la, lo],
            radius=6,
            color=color,
            weight=2,
            fill=True,
            fill_color=fill,
            fill_opacity=0.85,
            popup=folium.Popup(pop, max_width=340),
        ).add_to(fg_fac)
    fg_fac.add_to(fmap)

    _wm.add_match_lines_to_map(fmap, match_table, master.drop_duplicates(subset=["master_id"]))

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))
    print(f"  Wrote link map: {out_html}")


def _fmt_popup_kg(kg: object) -> str:
    try:
        x = float(kg)
        if math.isfinite(x):
            return html.escape(f"{x:.4g}")
    except (TypeError, ValueError):
        pass
    return html.escape(str(kg))


def main() -> None:
    root = _root()
    cd = _load_cams_d_map()
    ap = argparse.ArgumentParser(
        description=(
            "Match CAMS GNFR D point sources to F5_2 LCP (preferred) and F1_4 AIR facilities; "
            "write CSV + GeoTIFF; optional Folium HTML linking CAMS to facilities."
        )
    )
    ap.add_argument("--nc", type=Path, default=None, help="CAMS NetCDF (default: data/given_CAMS/.../CAMS-REG-v8_1_emissions_year2019.nc)")
    ap.add_argument("--country", default="GRC", help="ISO3 country filter for CAMS + E-PRTR")
    ap.add_argument("--all-countries", action="store_true", help="Use full CAMS + PRTR domain")
    ap.add_argument("--pollutant", default="nmvoc", help="NetCDF variable for emission mass (kg/yr) raster")
    ap.add_argument(
        "--eprtr-pollutant",
        default=cd.DEFAULT_EPRTR_POLLUTANT,
        help="F1_4 Pollutant column filter (exact string)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Fugitive/outputs/fugitive_point_proxy)",
    )
    ap.add_argument("--pixel-size-deg", type=float, default=0.01, help="GeoTIFF pixel size (degrees)")
    ap.add_argument(
        "--distance-scale",
        type=float,
        default=1.0,
        help="Scale max search radius (same meaning as Waste proxy matcher)",
    )
    ap.add_argument(
        "--tif-name",
        default="Fugitive_pointsource.tif",
        help="Output GeoTIFF filename inside --out-dir",
    )
    ap.add_argument(
        "--f5-csv",
        type=Path,
        default=None,
        help="F5_2 LCP CSV (default from cams_D defaults)",
    )
    ap.add_argument(
        "--f1-csv",
        type=Path,
        default=None,
        help="F1_4 facilities CSV (default from cams_D defaults)",
    )
    ap.add_argument(
        "--write-map",
        action="store_true",
        help="Write an interactive HTML map (CAMS points, E-PRTR facilities, polylines).",
    )
    ap.add_argument(
        "--out-map-html",
        type=Path,
        default=None,
        help="Output HTML path (default: <out-dir>/fugitive_cams_proxy_links.html)",
    )
    ap.add_argument("--map-zoom", type=int, default=7, help="Initial zoom for --write-map")
    args = ap.parse_args()

    nc = args.nc
    if nc is None:
        nc = cd.DEFAULT_NC
    elif not nc.is_absolute():
        nc = root / nc

    f5p = args.f5_csv if args.f5_csv is not None else cd.DEFAULT_EPRTR_F5_LCP_CSV
    if not f5p.is_absolute():
        f5p = root / f5p
    f1p = args.f1_csv if args.f1_csv is not None else cd.DEFAULT_EPRTR_FACILITIES_CSV
    if not f1p.is_absolute():
        f1p = root / f1p

    out_dir = args.out_dir if args.out_dir is not None else root / "Fugitive" / "outputs" / "fugitive_point_proxy"
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if not nc.is_file():
        raise SystemExit(f"NetCDF not found: {nc}")
    if not f5p.is_file():
        raise SystemExit(f"F5_2 CSV not found: {f5p}")
    if not f1p.is_file():
        raise SystemExit(f"F1_4 CSV not found: {f1p}")

    bbox_t = None
    master = build_fugitive_facility_master(
        country_arg=args.country,
        all_countries=args.all_countries,
        bbox=bbox_t,
        f5_csv=f5p,
        f1_csv=f1p,
        eprtr_pollutant=args.eprtr_pollutant,
        cd=cd,
    )
    if master.empty:
        raise SystemExit("No facilities in master (check country / CSVs).")

    cams = load_cams_d_points(
        nc,
        country_iso3=None if args.all_countries else args.country,
        all_countries=args.all_countries,
        bbox=bbox_t,
        pollutant=args.pollutant,
    )
    if cams.empty:
        raise SystemExit("No CAMS GNFR D point sources for this filter.")

    res = run_fugitive_proxy_workflow(
        master,
        cams,
        out_dir=out_dir,
        pixel_size_deg=float(args.pixel_size_deg),
        tif_filename=str(args.tif_name),
        distance_scale=float(args.distance_scale),
    )
    if args.write_map:
        map_path = args.out_map_html
        if map_path is None:
            map_path = out_dir / "fugitive_cams_proxy_links.html"
        elif not map_path.is_absolute():
            map_path = root / map_path
        build_fugitive_proxy_link_map_html(
            res["match_table"],
            master,
            map_path,
            map_zoom=int(args.map_zoom),
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
