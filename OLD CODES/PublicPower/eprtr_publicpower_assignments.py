from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "PublicPower" / "eptr_csv"
IED_CSV = DATA_DIR / "F6_1_IED_Installations.csv"
LCP_CSV = DATA_DIR / "F5_2_LCP_Energy_Emissions.csv"
FLAGS_JSON = ROOT / "urbem_interface" / "config" / "factory_bundled" / "eprtr_snap_assignment.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_annex_i_activity(value: object) -> str:
    """Keep this logic aligned with the existing proxy-factory parser."""
    if pd.isna(value) or not str(value).strip():
        return "0 -z"
    text = str(value).strip()
    if "," in text:
        text = text.split(",")[0].strip()
    sector = "0"
    subsector = "z"
    match = re.search(r"(\d+)\s*[.\-\s(]*(\d+)", text)
    if match:
        sector = match.group(1)[0]
        num = int(match.group(2))
        subsector = chr(ord("a") + num - 1) if 1 <= num <= 26 else "z"
    else:
        digits = re.findall(r"\d", text)
        if digits:
            sector = digits[0]
            if len(digits) >= 2 and 1 <= int(digits[1]) <= 9:
                subsector = chr(ord("a") + int(digits[1]) - 1)
        if subsector == "z" and any(ch.isalpha() for ch in text):
            for ch in text:
                if ch.isalpha():
                    subsector = ch.lower()
                    break
    return sector + " -" + subsector


def load_energy_activity_labels(path: Path = FLAGS_JSON) -> dict[tuple[str, str], str]:
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    labels: dict[tuple[str, str], str] = {}
    for rules in cfg.values():
        if not isinstance(rules, list):
            continue
        for rule in rules:
            sector = str(rule.get("sector", ""))
            subsector = str(rule.get("subsector", ""))
            description = str(rule.get("description", "")).strip()
            if sector == "1" and subsector:
                labels[(sector, subsector)] = description
    return labels


def infer_fuel_family_from_text(*values: object) -> str:
    text = " ".join(str(v) for v in values if isinstance(v, str)).lower()
    if not text.strip():
        return "unknown"
    rules = [
        ("waste_to_energy", ["waste", "inciner", "valorisation", "wte"]),
        ("biomass_biogas", ["biomass", "biogas", "bioenergy", "wood", "pellet"]),
        ("coal_lignite", ["coal", "lignite", "brown coal", "hard coal", "coke"]),
        ("gas", ["gas", "ccgt", "turbine", "lng"]),
        ("oil", ["oil", "diesel", "fuel oil", "mazut"]),
        ("chp_heat", ["chp", "cogeneration", "heizkraftwerk", "district heating", "heat plant"]),
    ]
    for family, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return family
    return "unknown"


def build_installation_activity_table(
    ied_df: pd.DataFrame,
    activity_labels: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    labels = activity_labels or {}
    out = ied_df.copy()
    out["geometry_layer"] = out["EPRTRAnnexIMainActivity"].map(parse_annex_i_activity).astype(str)
    out["Sector"] = out["geometry_layer"].str.get(0).astype(str)
    out["SubSector"] = out["geometry_layer"].str.get(3).astype(str)
    out["activity_label"] = [
        labels.get((sector, subsector), "unmapped")
        for sector, subsector in zip(out["Sector"], out["SubSector"])
    ]
    out["is_energy_activity"] = out["Sector"] == "1"
    out["is_public_power_candidate"] = (out["Sector"] == "1") & (out["SubSector"] == "c")
    out["text_fuel_hint"] = [
        infer_fuel_family_from_text(name, activity)
        for name, activity in zip(out.get("installationName", ""), out.get("IEDMainActivityName", ""))
    ]
    return out


def build_activity_summary(classified_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        classified_df[classified_df["Sector"] == "1"]
        .groupby(["Sector", "SubSector", "activity_label"], as_index=False)
        .agg(
            n_installations=("installationName", "size"),
            n_countries=("CountryName", "nunique"),
            n_public_power_candidates=("is_public_power_candidate", "sum"),
        )
        .sort_values(["Sector", "SubSector"])
    )
    return summary


def load_lcp_characteristics(path: Path = LCP_CSV) -> pd.DataFrame:
    lcp_df = pd.read_csv(path, sep=";", low_memory=False)
    keep = lcp_df[
        (lcp_df["featureType"].astype(str) == "LCPCharacteristics")
        & (lcp_df["unit"].astype(str) == "MW")
    ].copy()
    keep = keep.rename(columns={"countryName": "CountryName"})
    return keep[
        [
            "CountryName",
            "reportingYear",
            "LCPInspireId",
            "installationPartName",
            "City_Of_Facility",
            "Longitude",
            "Latitude",
            "featureValue",
        ]
    ]


def run(
    ied_csv: Path = IED_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ied_df = pd.read_csv(ied_csv, low_memory=False)
    classified_df = build_installation_activity_table(ied_df, load_energy_activity_labels())
    summary_df = build_activity_summary(classified_df)
    candidates_df = classified_df[classified_df["is_public_power_candidate"]].copy()

    candidates_df = candidates_df[
        [
            "CountryName",
            "reportingYear",
            "installationName",
            "City_of_Facility",
            "Longitude",
            "Latitude",
            "EPRTRAnnexIMainActivity",
            "IEDAnnexIMainActivity",
            "IEDMainActivityName",
            "Sector",
            "SubSector",
            "activity_label",
            "text_fuel_hint",
        ]
    ].sort_values(["CountryName", "installationName", "reportingYear"])

    summary_path = output_dir / "eprtr_energy_activity_summary.csv"
    candidates_path = output_dir / "public_power_1c_candidates.csv"
    summary_df.to_csv(summary_path, index=False)
    candidates_df.to_csv(candidates_path, index=False)

    outputs = {
        "activity_summary_csv": summary_path,
        "public_power_candidates_csv": candidates_path,
    }

    if LCP_CSV.is_file():
        lcp_df = load_lcp_characteristics(LCP_CSV)
        lcp_path = output_dir / "lcp_characteristics_mw.csv"
        lcp_df.to_csv(lcp_path, index=False)
        outputs["lcp_characteristics_csv"] = lcp_path

    return outputs


def main() -> None:
    outputs = run()
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
