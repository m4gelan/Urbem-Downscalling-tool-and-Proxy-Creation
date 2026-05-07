from __future__ import annotations

from pathlib import Path

import pandas as pd


def _weighted_intensity(process_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    merged = wide_df[["NUTS_ID", "CLC_CODE", "COUNTRY", "NAME_REGION", "n_pixels"]].merge(
        process_df[["NUTS_ID", "CLC_CODE", "mu", "rho"]],
        on=["NUTS_ID", "CLC_CODE"],
        how="left",
    )
    merged["mu"] = merged["mu"].fillna(0.0)
    merged["rho"] = merged["rho"].fillna(0.0)
    merged["weighted_mu"] = merged["mu"] * merged["n_pixels"]
    grouped = (
        merged.groupby(["NUTS_ID", "COUNTRY", "NAME_REGION"], as_index=False)
        .agg(
            ag_pixels=("n_pixels", "sum"),
            intensity_weighted_mu=("weighted_mu", "sum"),
            mean_rho=("rho", "mean"),
        )
    )
    grouped["ag_intensity_score"] = grouped["intensity_weighted_mu"] / grouped["ag_pixels"].where(
        grouped["ag_pixels"] > 0, 1.0
    )
    return grouped


def _class_share_table(wide_df: pd.DataFrame, intensity_df: pd.DataFrame) -> pd.DataFrame:
    shares = wide_df[["NUTS_ID", "COUNTRY", "NAME_REGION", "CLC_CODE", "n_pixels"]].copy()
    totals = shares.groupby("NUTS_ID")["n_pixels"].transform("sum")
    shares["baseline_share"] = shares["n_pixels"] / totals.where(totals > 0, 1.0)
    shares = shares.merge(
        intensity_df[["NUTS_ID", "ag_intensity_score", "mean_rho", "ag_pixels"]],
        on="NUTS_ID",
        how="left",
    )
    shares["selection_score"] = shares["baseline_share"] * shares["ag_intensity_score"]
    return shares


def _empty_selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country",
            "selection",
            "selection_rank",
            "NUTS_ID",
            "COUNTRY",
            "NAME_REGION",
            "ag_pixels",
            "ag_intensity_score",
            "mean_rho",
            "selected_clc_code",
            "selected_clc_share",
            "selected_clc_score",
            "process_id",
        ]
    )


def _filter_candidate_countries(class_df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    if not countries:
        return class_df.copy()
    keep = {str(country).strip().upper() for country in countries if str(country).strip()}
    if not keep:
        return class_df.copy()
    return class_df[class_df["COUNTRY"].astype(str).str.upper().isin(keep)].copy()


def _pick_regions_for_clc(
    clc_df: pd.DataFrame,
    *,
    regions_per_clc: int,
    used_regions: set[str],
) -> pd.DataFrame:
    ranked = clc_df.sort_values(
        ["selection_score", "baseline_share", "ag_intensity_score", "NUTS_ID"],
        ascending=[False, False, False, True],
    ).copy()

    chosen_rows: list[pd.Series] = []
    local_used: set[str] = set()
    target_n = max(1, int(regions_per_clc))

    for _, row in ranked.iterrows():
        nuts_id = str(row["NUTS_ID"])
        if nuts_id in used_regions or nuts_id in local_used:
            continue
        chosen_rows.append(row)
        local_used.add(nuts_id)
        if len(chosen_rows) >= target_n:
            break

    if len(chosen_rows) < target_n:
        for _, row in ranked.iterrows():
            nuts_id = str(row["NUTS_ID"])
            if nuts_id in local_used:
                continue
            chosen_rows.append(row)
            local_used.add(nuts_id)
            if len(chosen_rows) >= target_n:
                break

    out = pd.DataFrame(chosen_rows).copy()
    if out.empty:
        return out
    used_regions.update(out["NUTS_ID"].astype(str).tolist())
    out = out.rename(
        columns={
            "CLC_CODE": "selected_clc_code",
            "baseline_share": "selected_clc_share",
            "selection_score": "selected_clc_score",
        }
    )
    out["selection"] = "representative"
    out = out.sort_values(
        ["selected_clc_score", "selected_clc_share", "ag_intensity_score", "NUTS_ID"],
        ascending=[False, False, False, True],
    ).copy()
    out["selection_rank"] = range(1, len(out) + 1)
    return out


def _select_representative_regions_by_clc(
    class_df: pd.DataFrame,
    *,
    countries: list[str],
    process_id: str,
    regions_per_clc: int,
) -> pd.DataFrame:
    eligible = _filter_candidate_countries(class_df, countries)
    eligible = eligible[(eligible["ag_pixels"] > 0) & (eligible["n_pixels"] > 0)].copy()
    if eligible.empty:
        return _empty_selection_frame()

    frames: list[pd.DataFrame] = []
    used_regions: set[str] = set()
    class_priority = (
        eligible.groupby("CLC_CODE")["selection_score"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for clc_code in class_priority:
        clc_df = eligible[eligible["CLC_CODE"] == clc_code].copy()
        selected = _pick_regions_for_clc(
            clc_df,
            regions_per_clc=regions_per_clc,
            used_regions=used_regions,
        )
        if selected.empty:
            continue
        selected.insert(0, "country", selected["COUNTRY"].astype(str))
        selected["process_id"] = process_id
        frames.append(selected)

    if not frames:
        return _empty_selection_frame()

    out = pd.concat(frames, ignore_index=True)
    keep_cols = [
        "country",
        "selection",
        "selection_rank",
        "NUTS_ID",
        "COUNTRY",
        "NAME_REGION",
        "ag_pixels",
        "ag_intensity_score",
        "mean_rho",
        "selected_clc_code",
        "selected_clc_share",
        "selected_clc_score",
        "process_id",
    ]
    out = out[keep_cols].sort_values(
        ["selected_clc_code", "selection_rank", "selected_clc_score", "NUTS_ID"],
        ascending=[True, True, False, True],
    )
    return out.reset_index(drop=True)


def build_region_selection_from_tables(
    wide_df: pd.DataFrame,
    process_df: pd.DataFrame,
    countries: list[str],
    *,
    process_id: str = "fertilized_land",
    regions_per_clc: int = 4,
) -> pd.DataFrame:
    intensity_df = _weighted_intensity(process_df, wide_df)
    class_df = _class_share_table(wide_df, intensity_df)
    return _select_representative_regions_by_clc(
        class_df,
        countries=countries,
        process_id=process_id,
        regions_per_clc=regions_per_clc,
    )


def build_region_selection_table(
    result_root: Path,
    countries: list[str],
    *,
    process_id: str = "fertilized_land",
    regions_per_clc: int = 4,
) -> pd.DataFrame:
    wide_frames: list[pd.DataFrame] = []
    process_frames: list[pd.DataFrame] = []
    for country in countries:
        country_root = result_root / country
        wide_path = country_root / "weights_wide.csv"
        process_path = country_root / "intermediary" / f"rho_{process_id}.csv"
        if not wide_path.is_file():
            raise FileNotFoundError(f"Missing wide weights for {country}: {wide_path}")
        if not process_path.is_file():
            raise FileNotFoundError(f"Missing intermediary process table for {country}: {process_path}")
        wide_frames.append(pd.read_csv(wide_path))
        process_frames.append(pd.read_csv(process_path))
    if not wide_frames or not process_frames:
        return _empty_selection_frame()
    return build_region_selection_from_tables(
        pd.concat(wide_frames, ignore_index=True),
        pd.concat(process_frames, ignore_index=True),
        countries,
        process_id=process_id,
        regions_per_clc=regions_per_clc,
    )
