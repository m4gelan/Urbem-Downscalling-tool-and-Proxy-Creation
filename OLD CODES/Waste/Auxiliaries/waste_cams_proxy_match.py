"""
Empirical spatial proxy: match CAMS GNFR J (waste) point sources to nearby eligible
EEA / E-PRTR / IED facility coordinates. Not identity matching — scores and thresholds
are tunable constants below.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable rules (defaults per project brief)
# ---------------------------------------------------------------------------

# Synthetic category for EEA UWWTD treatment plants (GeoPackage points), same scoring pipeline as PRTR.
UWWTD_PROXY_CATEGORY = "uwwtd_treatment_plant"

# Eligible facility primary_category values only (matching candidate pool).
ELIGIBLE_CATEGORIES: tuple[str, ...] = (
    "waste_incineration",
    "waste_to_energy",
    "co_incineration",
    "landfill",
    "wastewater",
    "waste_air_release",
    UWWTD_PROXY_CATEGORY,
)

# Shown on map but never used as matching candidates.
NON_ELIGIBLE_REFERENCE_CATEGORIES: frozenset[str] = frozenset(
    {
        "ied_waste_installation",
        "waste_transfer",
        "other_potential_waste",
        "unclear_but_relevant",
    }
)

# Class priority index (lower = higher priority); used only for tie-break / ordering.
CLASS_PRIORITY_INDEX: dict[str, int] = {c: i for i, c in enumerate(ELIGIBLE_CATEGORIES)}

CLASS_SCORES: dict[str, float] = {
    "waste_incineration": 100.0,
    "waste_to_energy": 95.0,
    "co_incineration": 90.0,
    "landfill": 85.0,
    "wastewater": 80.0,
    "waste_air_release": 70.0,
    # Must clear ACCEPT_MIN_BEST_SCORE at max search distance (distance_score -> 0 at threshold km).
    UWWTD_PROXY_CATEGORY: 80.0,
}

MAX_DISTANCE_KM: dict[str, float] = {
    "waste_incineration": 5.0,
    "waste_to_energy": 6.0,
    "co_incineration": 5.0,
    "landfill": 7.0,
    "wastewater": 5.0,
    "waste_air_release": 7.0,
    UWWTD_PROXY_CATEGORY: 5.0,
}

# Acceptance: best total_score must reach this; if a 2nd candidate exists, gap must exceed margin.
ACCEPT_MIN_BEST_SCORE = 80.0
ACCEPT_MIN_MARGIN_VS_SECOND = 15.0

# Line styling (map only): green vs orange for "matched" polylines.
HIGH_MATCH_SCORE = 115.0
HIGH_MATCH_MARGIN = 20.0

# Evidence bonus caps (see evidence_bonus_for_facility).
EVIDENCE_BONUS_CAP = 25.0

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in km (vectorized)."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    o1 = np.radians(np.asarray(lon1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    o2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dp = p2 - p1
    do = o2 - o1
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(do / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def filter_eligible_matching_facilities(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty or "primary_category" not in master.columns:
        return pd.DataFrame()
    m = master["primary_category"].astype(str).isin(ELIGIBLE_CATEGORIES)
    m = m & ~master["primary_category"].astype(str).eq(UWWTD_PROXY_CATEGORY)
    out = master.loc[m].copy()
    return out


def uwwtd_points_df_to_proxy_facility_rows(uwwtd: pd.DataFrame | None) -> pd.DataFrame:
    """
    Build master-like rows from ``load_uwwtd_treatment_plants_gpkg`` output
    (``uwwtd_lat``, ``uwwtd_lon``, optional attribute columns) for the same candidate
    scoring as E-PRTR facilities. Max distance uses ``MAX_DISTANCE_KM`` for
    ``UWWTD_PROXY_CATEGORY`` (default 5 km, still multiplied by ``distance_scale``).
    """
    if uwwtd is None or uwwtd.empty:
        return pd.DataFrame()
    if "uwwtd_lat" not in uwwtd.columns or "uwwtd_lon" not in uwwtd.columns:
        return pd.DataFrame()
    name_hints = (
        "uwwtpName",
        "UWWTP_NAME",
        "name",
        "Name",
        "plantName",
        "TITLE",
        "title",
        "siteName",
        "facilityName",
    )
    rows: list[dict[str, Any]] = []
    for i, (_, r) in enumerate(uwwtd.iterrows()):
        mid = f"UWWTD_{i:06d}"
        nm = ""
        for c in name_hints:
            if c in uwwtd.columns and pd.notna(r.get(c)) and str(r.get(c)).strip():
                nm = str(r.get(c)).strip()
                break
        fid = ""
        for c in ("uwwtdReportedIdentifier", "eMsCode", "EU_CD", "NationalID", "nationalId"):
            if c in uwwtd.columns and pd.notna(r.get(c)) and str(r.get(c)).strip():
                fid = str(r.get(c)).strip()
                break
        rows.append(
            {
                "master_id": mid,
                "FacilityInspireId": fid,
                "facility_name": nm or mid,
                "longitude": float(r["uwwtd_lon"]),
                "latitude": float(r["uwwtd_lat"]),
                "primary_category": UWWTD_PROXY_CATEGORY,
                "from_f1_4": False,
                "from_f4_2": False,
                "from_f5_2": False,
                "from_f6_1": False,
                "from_f7_1": False,
                "has_air_release": False,
                "is_incinerator": False,
                "is_co_incinerator": False,
                "is_wastewater": True,
                "is_landfill": False,
                "is_wte": False,
            }
        )
    return pd.DataFrame(rows)


def resolve_uwwtd_ambiguous_nearest(
    g: pd.DataFrame,
    *,
    best_score: float,
    second_score: float,
) -> pd.Series | None:
    """
    When the score-based margin vs the 2nd candidate is too small but both top options
    are ``UWWTD_PROXY_CATEGORY``, pick the **nearest** plant (then ``facility_id``) so
    dense islands (e.g. multiple WWTP within a few km) still get a single proxy link.
    """
    if g.empty or len(g) < 2:
        return None
    if best_score < ACCEPT_MIN_BEST_SCORE:
        return None
    if best_score - second_score >= ACCEPT_MIN_MARGIN_VS_SECOND:
        return None
    g0 = g.iloc[0]
    g1 = g.iloc[1]
    if str(g0["facility_category"]) != UWWTD_PROXY_CATEGORY:
        return None
    if str(g1["facility_category"]) != UWWTD_PROXY_CATEGORY:
        return None
    uw = g[g["facility_category"].astype(str) == UWWTD_PROXY_CATEGORY]
    if uw.empty:
        return None
    mx = float(uw["total_score"].max())
    pool = uw[uw["total_score"] >= mx - ACCEPT_MIN_MARGIN_VS_SECOND].copy()
    if pool.empty:
        return None
    pick = pool.sort_values(["distance_km", "facility_id"], ascending=[True, True]).iloc[0]
    return pick


def distance_score_component(distance_km: float, class_threshold_km: float) -> float:
    if class_threshold_km <= 0:
        return 0.0
    t = 40.0 * (1.0 - distance_km / class_threshold_km)
    return float(np.clip(t, 0.0, 40.0))


def evidence_bonus_for_facility(row: pd.Series) -> float:
    """Optional +5..+15 for multi-source rows; +5 for strong incineration table signal."""
    if str(row.get("primary_category", "")) == UWWTD_PROXY_CATEGORY:
        return 0.0
    bonus = 0.0
    flags = [
        bool(row.get("from_f1_4")),
        bool(row.get("from_f4_2")),
        bool(row.get("from_f5_2")),
        bool(row.get("from_f6_1")),
        bool(row.get("from_f7_1")),
    ]
    n_src = sum(flags)
    if n_src >= 3:
        bonus += 15.0
    elif n_src == 2:
        bonus += 10.0
    elif n_src == 1:
        bonus += 5.0

    cat = str(row.get("primary_category", ""))
    if bool(row.get("from_f7_1")) and cat in ("waste_incineration", "co_incineration", "waste_to_energy"):
        bonus += 5.0

    eligible_hits = sum(
        1
        for k in (
            "has_air_release",
            "is_incinerator",
            "is_co_incinerator",
            "is_wastewater",
            "is_landfill",
            "is_wte",
        )
        if bool(row.get(k))
    )
    if eligible_hits >= 3:
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
    ev = evidence_bonus_for_facility(facility_row)
    total = base + dscore + ev
    return total, base, dscore, ev


def generate_candidate_pairs(
    cams: pd.DataFrame,
    facilities: pd.DataFrame,
    *,
    cams_lat_col: str = "cams_lat",
    cams_lon_col: str = "cams_lon",
    fac_lat_col: str = "latitude",
    fac_lon_col: str = "longitude",
    distance_scale: float = 1.0,
) -> pd.DataFrame:
    """
    One row per (CAMS point, facility) pair where distance <= class-specific max km
    (``MAX_DISTANCE_KM`` × ``distance_scale``).
    """
    if cams.empty or facilities.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    fac_lat = facilities[fac_lat_col].astype(np.float64).values
    fac_lon = facilities[fac_lon_col].astype(np.float64).values

    for ci, cr in cams.iterrows():
        sid = str(cr.get("cams_source_id", ""))
        if not sid:
            continue
        clat = float(cr[cams_lat_col])
        clon = float(cr[cams_lon_col])
        dist_all = haversine_km(clat, clon, fac_lat, fac_lon)

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


def assign_best_match(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank candidates per CAMS point; apply acceptance rule.
    Returns (match_table, candidates_with_rank).
    """
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
        if best_score < ACCEPT_MIN_BEST_SCORE:
            status = "unmatched_below_score_threshold"
            matched = False
        elif len(g) > 1 and margin < ACCEPT_MIN_MARGIN_VS_SECOND:
            pick = resolve_uwwtd_ambiguous_nearest(g, best_score=best_score, second_score=second_score)
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


def acceptance_at_candidate_index(
    g: pd.DataFrame,
    k: int,
) -> tuple[bool, str]:
    """
    Same acceptance rule as ``assign_best_match``, but treating ``g.iloc[k]`` as the
    proposed match (``g`` sorted by ``candidate_rank`` ascending; ``k`` is 0-based).
    """
    if g.empty or k < 0 or k >= len(g):
        return False, "unmatched_no_candidate"
    suffix = g.iloc[k:].reset_index(drop=True)
    best_score = float(suffix.iloc[0]["total_score"])
    has_next = len(suffix) > 1
    second_score = float(suffix.iloc[1]["total_score"]) if has_next else float("-inf")
    margin = best_score - second_score if has_next else float("inf")
    if best_score < ACCEPT_MIN_BEST_SCORE:
        return False, "unmatched_below_score_threshold"
    if has_next and margin < ACCEPT_MIN_MARGIN_VS_SECOND:
        pick = resolve_uwwtd_ambiguous_nearest(suffix, best_score=best_score, second_score=second_score)
        if pick is None:
            return False, "ambiguous_multiple_candidates"
        if str(pick["facility_id"]) == str(suffix.iloc[0]["facility_id"]):
            return True, "matched"
        return False, "ambiguous_multiple_candidates"
    return True, "matched"


def assign_row_from_candidate_rank(
    sid: str,
    g: pd.DataFrame,
    k: int,
    *,
    matched: bool,
    match_status: str,
) -> dict[str, Any]:
    """Build one assign_partial row; ``best_*`` fields always describe rank-1 candidate."""
    g0 = g.iloc[0]
    if matched:
        bk = g.iloc[k]
        best_score = float(bk["total_score"])
        has_next = k + 1 < len(g)
        second_score = float(g.iloc[k + 1]["total_score"]) if has_next else float("-inf")
        margin = best_score - second_score if has_next else float("inf")
        return {
            "cams_source_id": sid,
            "match_score": best_score,
            "second_best_score": second_score if has_next else float("nan"),
            "score_margin_vs_second": margin if has_next else float("nan"),
            "match_status": match_status,
            "matched_master_id": str(bk["facility_id"]),
            "matched_category": str(bk["facility_category"]),
            "distance_km": float(bk["distance_km"]),
            "best_master_id": str(g0["facility_id"]),
            "best_category": str(g0["facility_category"]),
            "best_distance_km": float(g0["distance_km"]),
            "candidate_rank1_class_score": float(g0["class_score"]),
            "candidate_rank1_distance_score": float(g0["distance_score"]),
            "candidate_rank1_evidence_bonus": float(g0["evidence_bonus"]),
        }
    best_score = float(g0["total_score"])
    has_next = len(g) > 1
    second_score = float(g.iloc[1]["total_score"]) if has_next else float("-inf")
    margin = best_score - second_score if has_next else float("inf")
    return {
        "cams_source_id": sid,
        "match_score": best_score,
        "second_best_score": second_score if has_next else float("nan"),
        "score_margin_vs_second": margin if has_next else float("nan"),
        "match_status": match_status,
        "matched_master_id": "",
        "matched_category": "",
        "distance_km": float("nan"),
        "best_master_id": str(g0["facility_id"]),
        "best_category": str(g0["facility_category"]),
        "best_distance_km": float(g0["distance_km"]),
        "candidate_rank1_class_score": float(g0["class_score"]),
        "candidate_rank1_distance_score": float(g0["distance_score"]),
        "candidate_rank1_evidence_bonus": float(g0["evidence_bonus"]),
    }


def _demote_exclusive_loss_row(template: pd.Series) -> dict[str, Any]:
    """Clear matched facility; keep rank-1 candidate diagnostics on ``best_*`` / ``candidate_rank1_*``."""
    sid = str(template.get("cams_source_id", ""))
    return {
        "cams_source_id": sid,
        "match_score": float("nan"),
        "second_best_score": float("nan"),
        "score_margin_vs_second": float("nan"),
        "match_status": "unmatched_facility_exclusive_loss",
        "matched_master_id": "",
        "matched_category": "",
        "distance_km": float("nan"),
        "best_master_id": str(template.get("best_master_id", "")),
        "best_category": str(template.get("best_category", "")),
        "best_distance_km": float(template.get("best_distance_km", float("nan"))),
        "candidate_rank1_class_score": float(template.get("candidate_rank1_class_score", float("nan"))),
        "candidate_rank1_distance_score": float(template.get("candidate_rank1_distance_score", float("nan"))),
        "candidate_rank1_evidence_bonus": float(template.get("candidate_rank1_evidence_bonus", float("nan"))),
    }


def _apply_row_dict(out: pd.DataFrame, idx: Any, row_dict: dict[str, Any]) -> None:
    for col, val in row_dict.items():
        if col in out.columns:
            out.loc[idx, col] = val


def resolve_unique_facility_per_match(
    assign_partial: pd.DataFrame,
    cand_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enforce at most one CAMS point per matched facility.

    When several CAMS rows are ``matched`` to the same ``matched_master_id``, keep the
    link with highest ``match_score`` (tie-break: lexicographic ``cams_source_id``).
    Each other CAMS point is demoted and may only fall back to its **second-ranked**
    candidate (same score/margin acceptance rules as the primary match). If that
    candidate is missing, fails acceptance, or its facility is already claimed by
    another matched CAMS point, status becomes ``unmatched_facility_exclusive_loss``.
    """
    if assign_partial.empty:
        return assign_partial
    out = assign_partial.copy()
    if cand_ranked.empty:
        return out

    cand_by_sid: dict[str, pd.DataFrame] = {
        str(sid): g.sort_values("candidate_rank", ascending=True).reset_index(drop=True)
        for sid, g in cand_ranked.groupby("cams_source_id", sort=False)
    }

    def claimed_master_ids() -> set[str]:
        m = out["match_status"].astype(str) == "matched"
        if not m.any():
            return set()
        mids = out.loc[m, "matched_master_id"].astype(str)
        return {x for x in mids if x}

    max_outer = max(len(out), 8) + 2
    for _ in range(max_outer):
        m_ok = out["match_status"].astype(str) == "matched"
        if not m_ok.any():
            break
        sub = out.loc[m_ok, ["cams_source_id", "matched_master_id", "match_score"]].copy()
        vc = sub["matched_master_id"].astype(str).value_counts()
        dup_mids = vc[vc > 1].index.tolist()
        if not dup_mids:
            break

        loser_meta: list[tuple[str, float]] = []
        for mid in dup_mids:
            part = sub[sub["matched_master_id"].astype(str) == mid].copy()
            part = part.sort_values(["match_score", "cams_source_id"], ascending=[False, True])
            for _, rr in part.iloc[1:].iterrows():
                lsid = str(rr["cams_source_id"])
                loser_meta.append((lsid, float(rr["match_score"])))

        seen: set[str] = set()
        ordered_losers: list[tuple[str, float]] = []
        for lsid, sc in sorted(loser_meta, key=lambda t: (-t[1], t[0])):
            if lsid in seen:
                continue
            seen.add(lsid)
            ordered_losers.append((lsid, sc))

        for lsid, _ in ordered_losers:
            row_m = out["cams_source_id"].astype(str) == lsid
            if not row_m.any():
                continue
            idx = out.index[row_m][0]
            _apply_row_dict(out, idx, _demote_exclusive_loss_row(out.loc[idx]))

        claimed = claimed_master_ids()
        for lsid, _prev_sc in sorted(ordered_losers, key=lambda t: (-t[1], t[0])):
            row_m = out["cams_source_id"].astype(str) == lsid
            if not row_m.any():
                continue
            idx = out.index[row_m][0]
            g = cand_by_sid.get(lsid)
            if g is None or len(g) < 2:
                continue

            k = 1
            ok, _st = acceptance_at_candidate_index(g, k)
            if ok:
                fac = str(g.iloc[k]["facility_id"])
                if fac not in claimed:
                    new_row = assign_row_from_candidate_rank(lsid, g, k, matched=True, match_status="matched")
                    _apply_row_dict(out, idx, new_row)
                    claimed.add(fac)

    return out


def build_match_tables(
    cams: pd.DataFrame,
    facilities: pd.DataFrame,
    candidates: pd.DataFrame,
    assign_partial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join CAMS coords / names onto assign_partial -> full match_table + diagnostics."""
    cams_cols = ["cams_source_id", "cams_lon", "cams_lat"]
    if "cams_pollutant_kg" in cams.columns:
        cams_cols.append("cams_pollutant_kg")
    cams_key = cams[cams_cols].drop_duplicates(subset=["cams_source_id"], keep="first")
    out = assign_partial.merge(cams_key, on="cams_source_id", how="left")

    fac_by_mid = facilities.drop_duplicates(subset=["master_id"]).set_index("master_id")

    def fac_row(mid: str) -> pd.Series | None:
        if not mid or mid not in fac_by_mid.index:
            return None
        r = fac_by_mid.loc[mid]
        return r if isinstance(r, pd.Series) else r.iloc[0]

    def fac_name(mid: str) -> str:
        r = fac_row(mid)
        return str(r["facility_name"]) if r is not None else ""

    def fac_lonlat(mid: str) -> tuple[float, float]:
        r = fac_row(mid)
        if r is None:
            return float("nan"), float("nan")
        return float(r["longitude"]), float(r["latitude"])

    def fac_inspire(mid: str) -> str:
        r = fac_row(mid)
        return str(r.get("FacilityInspireId", "") or "") if r is not None else ""

    mlon, mlat = [], []
    for _, r in out.iterrows():
        lo, la = fac_lonlat(str(r.get("matched_master_id", "")))
        mlon.append(lo)
        mlat.append(la)
    out["matched_lon"] = mlon
    out["matched_lat"] = mlat
    out["matched_facility_name"] = [fac_name(str(x)) for x in out["matched_master_id"]]
    out["matched_facility_id"] = [fac_inspire(str(x)) for x in out["matched_master_id"]]

    st = out["match_status"].astype(str)
    out["used_original_cams_coordinate"] = ~st.eq("matched")

    # Diagnostics: candidates with rank
    diag = candidates.copy()
    return out, diag


def build_relocated_point_proxy(match_table: pd.DataFrame, emission_col: str = "cams_pollutant_kg") -> pd.DataFrame:
    """One row per CAMS point with final proxy lon/lat and emission weight."""
    rows: list[dict[str, Any]] = []
    for _, r in match_table.iterrows():
        sid = str(r["cams_source_id"])
        em = r.get(emission_col)
        try:
            w = float(em)
        except (TypeError, ValueError):
            w = float("nan")
        if not math.isfinite(w) or w < 0:
            w = 0.0

        if str(r.get("match_status")) == "matched":
            plon = float(r["matched_lon"])
            plat = float(r["matched_lat"])
        else:
            plon = float(r["cams_lon"])
            plat = float(r["cams_lat"])

        rows.append(
            {
                "cams_source_id": sid,
                "proxy_lon": plon,
                "proxy_lat": plat,
                "emission_kg": w,
                "match_status": str(r.get("match_status", "")),
            }
        )
    return pd.DataFrame(rows)


def rasterize_proxy_points_to_tif(
    proxy_df: pd.DataFrame,
    out_path: Path,
    *,
    pixel_size_deg: float = 0.01,
    margin_deg: float = 0.02,
    crs: str = "EPSG:4326",
) -> None:
    """Sum emissions (kg/yr) per cell; single-band Float32 GeoTIFF, 0 where empty."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise SystemExit(
            "rasterio is required for GeoTIFF export. Install with: pip install rasterio"
        ) from exc

    if proxy_df.empty:
        raise ValueError("proxy_df is empty; cannot rasterize.")

    lons = proxy_df["proxy_lon"].astype(np.float64).values
    lats = proxy_df["proxy_lat"].astype(np.float64).values
    wts = proxy_df["emission_kg"].astype(np.float64).values

    m = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(wts)
    lons, lats, wts = lons[m], lats[m], wts[m]
    if lons.size == 0:
        raise ValueError("No valid proxy coordinates for raster.")

    west = float(np.min(lons)) - margin_deg
    east = float(np.max(lons)) + margin_deg
    south = float(np.min(lats)) - margin_deg
    north = float(np.max(lats)) + margin_deg

    px = float(pixel_size_deg)
    width = max(1, int(math.ceil((east - west) / px)))
    height = max(1, int(math.ceil((north - south) / px)))
    transform = from_bounds(west, south, east, north, width, height)

    acc = np.zeros((height, width), dtype=np.float64)
    for lon, lat, wt in zip(lons, lats, wts):
        if not math.isfinite(wt) or wt < 0:
            continue
        col = int((lon - west) / px)
        row = int((north - lat) / px)
        if 0 <= row < height and 0 <= col < width:
            acc[row, col] += wt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": None,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(acc.astype(np.float32), 1)
        dst.set_band_description(1, "sum_CAMS_proxy_pollutant_kg_per_cell")


def facility_cams_match_counts(match_table: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    sub = match_table[match_table["match_status"].astype(str) == "matched"]
    for mid in sub["matched_master_id"].astype(str):
        if not mid:
            continue
        counts[mid] = counts.get(mid, 0) + 1
    return counts


def match_line_style(match_table_row: pd.Series) -> dict[str, Any]:
    """Folium PolyLine kwargs (CAMS original -> facility or best candidate)."""
    st = str(match_table_row.get("match_status", ""))
    if st == "ambiguous_multiple_candidates":
        return {"color": "#d73027", "weight": 2, "opacity": 0.75, "dash_array": "6,4"}
    if st != "matched":
        return {}
    score = float(match_table_row.get("match_score", 0.0))
    margin = float(match_table_row.get("score_margin_vs_second", float("nan")))
    if score >= HIGH_MATCH_SCORE and (
        math.isinf(margin) or (math.isfinite(margin) and margin >= HIGH_MATCH_MARGIN)
    ):
        return {"color": "#1a9850", "weight": 3, "opacity": 0.85}
    return {"color": "#f46d43", "weight": 2, "opacity": 0.8}


def add_match_lines_to_map(
    fmap: Any,
    match_table: pd.DataFrame,
    facility_lookup: pd.DataFrame,
) -> None:
    """Draw Polylines from original CAMS coordinates to matched (or best) facility coordinates."""
    import folium

    if match_table.empty:
        return
    mid_idx = facility_lookup.drop_duplicates(subset=["master_id"]).set_index("master_id")
    fg = folium.FeatureGroup(name="CAMS–facility proxy links", show=True)
    for _, r in match_table.iterrows():
        st = str(r.get("match_status", ""))
        if st not in ("matched", "ambiguous_multiple_candidates"):
            continue
        try:
            clat = float(r["cams_lat"])
            clon = float(r["cams_lon"])
        except (TypeError, ValueError):
            continue
        if st == "matched":
            try:
                elat = float(r["matched_lat"])
                elon = float(r["matched_lon"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(elat) and math.isfinite(elon)):
                continue
        else:
            bid = str(r.get("best_master_id", ""))
            if not bid or bid not in mid_idx.index:
                continue
            rr = mid_idx.loc[bid]
            if isinstance(rr, pd.DataFrame):
                rr = rr.iloc[0]
            elat = float(rr["latitude"])
            elon = float(rr["longitude"])
        style = match_line_style(r)
        if not style:
            continue
        folium.PolyLine(
            locations=[(clat, clon), (elat, elon)],
            **style,
        ).add_to(fg)
    fg.add_to(fmap)


def run_proxy_workflow(
    master: pd.DataFrame,
    cams: pd.DataFrame,
    *,
    out_dir: Path,
    pixel_size_deg: float,
    tif_filename: str = "Waste_pointsource.tif",
    distance_scale: float = 1.0,
    uwwtd_points: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Run matching, write CSVs, write GeoTIFF, return objects for Folium integration.

    Expects ``cams`` with columns: cams_source_id, cams_lon, cams_lat, cams_pollutant_kg.
    Expects ``master`` with primary_category, master_id, facility columns + from_* flags.

    ``distance_scale`` multiplies every entry in ``MAX_DISTANCE_KM`` (search radius only;
    class scores and acceptance thresholds are unchanged unless you edit those constants).

    ``uwwtd_points``: optional output of ``load_uwwtd_treatment_plants_gpkg``; appended as
    candidates with category ``uwwtd_treatment_plant`` (default 5 km cap).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dscale = float(distance_scale)
    if dscale <= 0:
        raise ValueError("distance_scale must be positive.")

    elig = filter_eligible_matching_facilities(master)
    uw_rows = uwwtd_points_df_to_proxy_facility_rows(
        uwwtd_points if uwwtd_points is not None else pd.DataFrame()
    )
    fac_pool = (
        pd.concat([elig, uw_rows], ignore_index=True)
        if not uw_rows.empty
        else elig.copy()
    )
    print(
        f"CAMS–facility proxy: {len(cams)} CAMS points, "
        f"{len(elig)} eligible PRTR/IED facilities"
        + (f", +{len(uw_rows)} UWWTD plants (≤{MAX_DISTANCE_KM[UWWTD_PROXY_CATEGORY] * dscale:g} km)" if not uw_rows.empty else "")
        + "."
    )
    if dscale != 1.0:
        print(f"  Distance scale: {dscale:g} (max km per class = table × {dscale:g})")

    candidates = generate_candidate_pairs(cams, fac_pool, distance_scale=dscale)
    print(f"  Candidate pairs (within class distance thresholds): {len(candidates)}")

    assign_partial, cand_ranked = assign_best_match(candidates)
    # Re-append rows for CAMS ids with zero candidates
    known_ids = set(cams["cams_source_id"].astype(str))
    assigned_ids = set(assign_partial["cams_source_id"].astype(str)) if not assign_partial.empty else set()
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

    assign_partial = resolve_unique_facility_per_match(assign_partial, cand_ranked)
    n_excl = int((assign_partial["match_status"].astype(str) == "unmatched_facility_exclusive_loss").sum())
    if n_excl:
        print(
            f"  Facility-exclusive rule: {n_excl} CAMS point(s) unmatched "
            "(lost duplicate facility claim; no acceptable 2nd-ranked proxy)."
        )

    match_table, diag = build_match_tables(cams, fac_pool, candidates, assign_partial)

    # Counts
    vc = match_table["match_status"].value_counts()
    print("  Match status counts:")
    for k, v in vc.items():
        print(f"    {k}: {int(v)}")
    n_cams = len(match_table)
    n_matched = int((match_table["match_status"].astype(str) == "matched").sum())
    n_not_attr = n_cams - n_matched
    if n_cams:
        print(
            f"  CAMS attributed to a proxy facility (matched): {n_matched} / {n_cams} "
            f"({100.0 * n_matched / n_cams:.1f}%)."
        )
        print(
            f"  CAMS not attributed to another point (unmatched or ambiguous): {n_not_attr} / {n_cams} "
            f"({100.0 * n_not_attr / n_cams:.1f}%)."
        )
    else:
        print("  CAMS attributed / not attributed: no CAMS points in match table.")

    match_path = out_dir / "waste_cams_facility_proxy_matches.csv"
    cand_path = out_dir / "waste_cams_facility_proxy_candidates.csv"
    match_table.to_csv(match_path, index=False)
    cand_ranked.to_csv(cand_path, index=False)
    print(f"  Wrote {match_path.name} ({len(match_table)} rows)")
    print(f"  Wrote {cand_path.name} ({len(cand_ranked)} rows)")

    proxy_df = build_relocated_point_proxy(match_table)
    tif_path = out_dir / tif_filename
    rasterize_proxy_points_to_tif(proxy_df, tif_path, pixel_size_deg=pixel_size_deg)
    print(f"  Wrote GeoTIFF {tif_path} (EPSG:4326, pixel {pixel_size_deg} deg, band = summed kg/yr).")

    fcounts = facility_cams_match_counts(match_table)

    # Column order for downstream / spec clarity
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


def cams_popup_html_with_match(base_popup: str, match_row: pd.Series | None) -> str:
    """Append match summary to existing CAMS popup HTML."""
    if match_row is None:
        return base_popup
    sid = html.escape(str(match_row.get("cams_source_id", "")))
    st = html.escape(str(match_row.get("match_status", "")))
    ms = match_row.get("match_score")
    ms_s = html.escape(f"{ms:.2f}" if isinstance(ms, (int, float)) and math.isfinite(float(ms)) else str(ms))
    dk = match_row.get("distance_km")
    dk_s = html.escape(f"{dk:.3f}" if isinstance(dk, (int, float)) and math.isfinite(float(dk)) else str(dk))
    mf = html.escape(str(match_row.get("matched_facility_id", "")))
    mm = html.escape(str(match_row.get("matched_master_id", "")))
    mn = html.escape(str(match_row.get("matched_facility_name", "")))
    mc = html.escape(str(match_row.get("matched_category", "")))
    orig = str(match_row.get("cams_lon", "")), str(match_row.get("cams_lat", ""))
    block = (
        "<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>"
        "<b>CAMS proxy match</b><br/>"
        f"<b>cams_source_id</b> {sid}<br/>"
        f"<b>Original CAMS</b> lon {html.escape(orig[0])}, lat {html.escape(orig[1])}<br/>"
        f"<b>match_status</b> {st}<br/>"
        f"<b>match_score</b> {ms_s} &nbsp; <b>distance_km</b> {dk_s}<br/>"
        f"<b>matched_facility_id</b> {mf}<br/>"
        f"<b>matched_master_id</b> {mm}<br/>"
        f"<b>matched_facility_name</b> {mn}<br/>"
        f"<b>matched_category</b> {mc}<br/>"
        f"<b>used_original_cams_coordinate</b> "
        f"{html.escape(str(match_row.get('used_original_cams_coordinate', '')))}<br/>"
    )
    if base_popup.strip():
        return base_popup + block
    return f"<div style='min-width:260px;font-size:12px'>{block}</div>"


def facility_popup_append_match_count(base_popup: str, master_id: str, counts: dict[str, int]) -> str:
    n = int(counts.get(str(master_id), 0))
    extra = (
        f"<hr style='margin:6px 0;border:none;border-top:1px solid #ccc'/>"
        f"<b>CAMS proxy matches</b> {n} point source(s) linked to this facility.<br/>"
    )
    return base_popup + extra
