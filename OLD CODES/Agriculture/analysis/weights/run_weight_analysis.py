from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Agriculture.analysis.visualization.plot_weight_distributions import (
    plot_clc_pollutant_dumbbells,
    plot_region_redistribution_heatmaps,
)
from Agriculture.analysis.weights.region_selection import (
    build_region_selection_from_tables,
    build_region_selection_table,
)
from Agriculture.config import load_agriculture_config, project_root
from Agriculture.core.pipeline import apply_runtime_overrides, run_pipeline_with_config

DEFAULT_COUNTRIES = [
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "EL",
    "ES",
    "FI",
    "FR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
]
DEFAULT_REGIONS_PER_CLC = 4
DEFAULT_PLOT_POLLUTANTS = ["CH4", "NH3", "NOx", "CO2", "NMVOC", "PM2.5"]


def _country_result_dir(root: Path, country: str) -> Path:
    return root / "Agriculture" / "results" / "analysis" / country


def _country_outputs_ready(root: Path, country: str) -> bool:
    country_dir = _country_result_dir(root, country)
    required = [
        country_dir / "weights_long.csv",
        country_dir / "weights_wide.csv",
        country_dir / "intermediary" / "rho_fertilized_land.csv",
    ]
    return all(path.is_file() for path in required)


def _main_outputs_ready(root: Path) -> bool:
    required = [
        root / "Agriculture" / "results" / "weights_long.csv",
        root / "Agriculture" / "results" / "weights_wide.csv",
        root / "Agriculture" / "results" / "intermediary" / "rho_fertilized_land.csv",
    ]
    return all(path.is_file() for path in required)


def _ensure_country_runs(cfg: dict, countries: list[str], *, root: Path) -> None:
    if _main_outputs_ready(root):
        print(f"Reusing current main results under: {root / 'Agriculture' / 'results'}")
        return
    for country in countries:
        if _country_outputs_ready(root, country):
            print(f"Reusing existing analysis outputs for {country}: {_country_result_dir(root, country)}")
            continue
        runtime = apply_runtime_overrides(
            cfg,
            country_override=country,
            outputs_override_dir=_country_result_dir(root, country),
            enable_visualization=False,
            intermediary=True,
        )
        run_pipeline_with_config(runtime)


def _selected_pairs(selection_df: pd.DataFrame) -> pd.DataFrame:
    return selection_df[
        [
            "country",
            "NUTS_ID",
            "selected_clc_code",
            "selection",
            "selection_rank",
            "ag_intensity_score",
            "selected_clc_share",
            "selected_clc_score",
        ]
    ].rename(columns={"selected_clc_code": "CLC_CODE"})


def _selected_regions(selection_df: pd.DataFrame) -> pd.DataFrame:
    return (
        selection_df[["country", "COUNTRY", "NUTS_ID", "NAME_REGION"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _build_comparison_table(result_root: Path, selection_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pairs = _selected_pairs(selection_df)
    for country in pairs["country"].dropna().astype(str).unique():
        long_path = result_root / country / "weights_long.csv"
        if not long_path.is_file():
            raise FileNotFoundError(f"Missing long weights for {country}: {long_path}")
        long_df = pd.read_csv(long_path)
        long_df["country"] = long_df["COUNTRY"].astype(str)
        denom = long_df.groupby(["pollutant", "NUTS_ID"])["n_pixels"].transform("sum")
        long_df["baseline_w"] = long_df["n_pixels"] / denom.where(denom > 0, 1.0)
        sub = long_df.merge(
            pairs[pairs["country"].astype(str) == country],
            on=["country", "NUTS_ID", "CLC_CODE"],
            how="inner",
        )
        sub["delta"] = sub["w_p"] - sub["baseline_w"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_comparison_from_main_results(root: Path, selection_df: pd.DataFrame) -> pd.DataFrame:
    long_path = root / "Agriculture" / "results" / "weights_long.csv"
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing long weights: {long_path}")
    long_df = pd.read_csv(long_path)
    long_df["country"] = long_df["COUNTRY"].astype(str)
    denom = long_df.groupby(["pollutant", "NUTS_ID"])["n_pixels"].transform("sum")
    long_df["baseline_w"] = long_df["n_pixels"] / denom.where(denom > 0, 1.0)
    sub = long_df.merge(
        _selected_pairs(selection_df),
        on=["country", "NUTS_ID", "CLC_CODE"],
        how="inner",
    )
    sub["delta"] = sub["w_p"] - sub["baseline_w"]
    return sub


def _build_region_redistribution_from_long(long_df: pd.DataFrame, selection_df: pd.DataFrame) -> pd.DataFrame:
    long_df = long_df.copy()
    long_df["country"] = long_df["COUNTRY"].astype(str)
    denom = long_df.groupby(["pollutant", "NUTS_ID"])["n_pixels"].transform("sum")
    long_df["baseline_w"] = long_df["n_pixels"] / denom.where(denom > 0, 1.0)
    long_df["delta"] = long_df["w_p"] - long_df["baseline_w"]

    region_df = _selected_regions(selection_df)
    sub = long_df.merge(region_df, on=["country", "COUNTRY", "NUTS_ID", "NAME_REGION"], how="inner")
    sub["class_share"] = sub["baseline_w"]
    return sub


def _build_region_redistribution_table(result_root: Path, selection_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    region_df = _selected_regions(selection_df)
    for country in region_df["country"].dropna().astype(str).unique():
        long_path = result_root / country / "weights_long.csv"
        if not long_path.is_file():
            raise FileNotFoundError(f"Missing long weights for {country}: {long_path}")
        long_df = pd.read_csv(long_path)
        frames.append(
            _build_region_redistribution_from_long(
                long_df,
                selection_df[selection_df["country"].astype(str) == country],
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_region_redistribution_from_main_results(root: Path, selection_df: pd.DataFrame) -> pd.DataFrame:
    long_path = root / "Agriculture" / "results" / "weights_long.csv"
    if not long_path.is_file():
        raise FileNotFoundError(f"Missing long weights: {long_path}")
    long_df = pd.read_csv(long_path)
    return _build_region_redistribution_from_long(long_df, selection_df)


def run_weight_analysis(
    config_path: Path | None = None,
    alpha_path: Path | None = None,
    *,
    countries: list[str] | None = None,
    regions_per_clc: int = DEFAULT_REGIONS_PER_CLC,
    plot_pollutants: list[str] | None = None,
) -> dict[str, str]:
    cfg = load_agriculture_config(config_path, alpha_path=alpha_path)
    root = project_root(cfg)
    chosen_countries = countries or DEFAULT_COUNTRIES
    chosen_pollutants = plot_pollutants or DEFAULT_PLOT_POLLUTANTS

    result_root = root / "Agriculture" / "results" / "analysis"
    result_root.mkdir(parents=True, exist_ok=True)

    _ensure_country_runs(cfg, chosen_countries, root=root)
    if _main_outputs_ready(root):
        selection_df = build_region_selection_from_tables(
            pd.read_csv(root / "Agriculture" / "results" / "weights_wide.csv"),
            pd.read_csv(root / "Agriculture" / "results" / "intermediary" / "rho_fertilized_land.csv"),
            chosen_countries,
            process_id="fertilized_land",
            regions_per_clc=regions_per_clc,
        )
        comparison_df = _build_comparison_from_main_results(root, selection_df)
        redistribution_df = _build_region_redistribution_from_main_results(root, selection_df)
    else:
        selection_df = build_region_selection_table(
            result_root,
            chosen_countries,
            process_id="fertilized_land",
            regions_per_clc=regions_per_clc,
        )
        comparison_df = _build_comparison_table(result_root, selection_df)
        redistribution_df = _build_region_redistribution_table(result_root, selection_df)

    selection_path = result_root / "selected_regions.csv"
    selection_df.to_csv(selection_path, index=False)

    comparison_path = result_root / "baseline_vs_upgraded.csv"
    comparison_df.to_csv(comparison_path, index=False)

    redistribution_path = result_root / "region_redistribution.csv"
    redistribution_df.to_csv(redistribution_path, index=False)

    plots_root = result_root / "plots"
    plot_clc_pollutant_dumbbells(
        comparison_df,
        chosen_pollutants,
        plots_root / "clc_pollutant_dumbbells",
    )
    plot_region_redistribution_heatmaps(
        redistribution_df,
        chosen_pollutants,
        plots_root / "region_redistribution_heatmaps",
        min_share=0.025,
    )

    return {
        "selection_csv": str(selection_path),
        "comparison_csv": str(comparison_path),
        "redistribution_csv": str(redistribution_path),
        "plots_dir": str(plots_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--alpha", type=Path, default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--regions-per-clc", type=int, default=DEFAULT_REGIONS_PER_CLC)
    parser.add_argument("--plot-pollutants", nargs="*", default=None)
    args = parser.parse_args()
    result = run_weight_analysis(
        config_path=args.config,
        alpha_path=args.alpha,
        countries=args.countries,
        regions_per_clc=args.regions_per_clc,
        plot_pollutants=args.plot_pollutants,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
