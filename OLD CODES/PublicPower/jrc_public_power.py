from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JRC_DIR = ROOT / "data" / "PublicPower" / "JRC"
JRC_UNITS_CSV = JRC_DIR / "JRC_OPEN_UNITS.csv"
DATAPACKAGE_JSON = JRC_DIR / "datapackage.json"
EF_JSON = Path(__file__).resolve().parent / "config" / "public_power_emission_factors.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

EIC_PATTERN = re.compile(r"^[A-Za-z0-9\-]{16}$")

TYPE_G_ALIASES: dict[str, str] = {
    "fossil hard coal": "Fossil Hard coal",
    "fossil gas": "Fossil Gas",
}

COMBUSTION_TYPES = frozenset(
    {
        "Fossil Hard coal",
        "Fossil Brown coal/Lignite",
        "Fossil Peat",
        "Fossil Oil",
        "Fossil Oil shale",
        "Fossil Gas",
        "Fossil Coal-derived gas",
        "Biomass",
        "Waste",
    }
)


def load_datapackage_type_g_enum(path: Path = DATAPACKAGE_JSON) -> frozenset[str]:
    with path.open(encoding="utf-8") as handle:
        dp = json.load(handle)
    for res in dp.get("resources", []):
        if res.get("name") != "units":
            continue
        for field in res.get("schema", {}).get("fields", []):
            if field.get("name") != "type_g":
                continue
            enum = field.get("constraints", {}).get("enum")
            if enum:
                return frozenset(enum)
    raise ValueError("type_g enum not found in datapackage.json")


def normalize_type_g(raw: object, canonical: frozenset[str]) -> str:
    if pd.isna(raw) or raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    key = s.lower()
    if key in TYPE_G_ALIASES:
        return TYPE_G_ALIASES[key]
    if s in canonical:
        return s
    for c in canonical:
        if s.lower() == c.lower():
            return c
    return s


def validate_eic(value: object) -> bool:
    if pd.isna(value) or value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    return bool(EIC_PATTERN.fullmatch(s))


def validate_jrc_open_units(
    df: pd.DataFrame,
    canonical_type_g: frozenset[str] | None = None,
) -> dict[str, Any]:
    canonical_type_g = canonical_type_g or load_datapackage_type_g_enum()
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_missing_lat": int(df["lat"].isna().sum()) if "lat" in df.columns else None,
        "n_missing_lon": int(df["lon"].isna().sum()) if "lon" in df.columns else None,
        "n_invalid_lat": 0,
        "n_invalid_lon": 0,
        "n_invalid_eic_p": 0,
        "n_invalid_eic_g": 0,
        "n_negative_capacity_g": 0,
        "unknown_type_g_after_normalize": [],
        "type_g_value_counts_normalized": {},
    }
    if "lat" in df.columns:
        lat = df["lat"]
        report["n_invalid_lat"] = int(((lat < -90) | (lat > 90)).fillna(False).sum())
    if "lon" in df.columns:
        lon = df["lon"]
        report["n_invalid_lon"] = int(((lon < -180) | (lon > 180)).fillna(False).sum())
    if "eic_p" in df.columns:
        report["n_invalid_eic_p"] = int((~df["eic_p"].map(validate_eic)).sum())
    if "eic_g" in df.columns:
        report["n_invalid_eic_g"] = int((~df["eic_g"].map(validate_eic)).sum())
    if "capacity_g" in df.columns:
        report["n_negative_capacity_g"] = int((df["capacity_g"] < 0).sum())

    norm = df["type_g"].map(lambda x: normalize_type_g(x, canonical_type_g)) if "type_g" in df.columns else pd.Series(dtype=object)
    unknown = sorted({x for x in norm.unique() if x and x not in canonical_type_g})
    report["unknown_type_g_after_normalize"] = unknown
    vc = norm.value_counts()
    report["type_g_value_counts_normalized"] = {str(k): int(v) for k, v in vc.items()}
    return report


def load_emission_factor_config(path: Path = EF_JSON) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def apply_proxy_weights(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    weights: dict[str, dict[str, float]] = cfg.get("proxy_weight", {})
    pollutants = ["nox", "sox", "nh3", "pm2_5", "pm10", "co", "nmvoc", "ch4"]
    out = df.copy()
    canonical = load_datapackage_type_g_enum()
    tg = out["type_g_normalized"] if "type_g_normalized" in out.columns else out["type_g"].map(lambda x: normalize_type_g(x, canonical))

    default_row = weights.get("Other", {})
    for pol in pollutants:
        col = f"proxy_w_{pol}"
        out[col] = tg.map(lambda t: float(weights.get(t, default_row).get(pol, 0.0)))

    out["is_combustion_generation"] = tg.isin(COMBUSTION_TYPES)
    return out


def prepare_jrc_units_table(
    csv_path: Path = JRC_UNITS_CSV,
    ef_path: Path = EF_JSON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    canonical = load_datapackage_type_g_enum()
    df["type_g_normalized"] = df["type_g"].map(lambda x: normalize_type_g(x, canonical))
    report = validate_jrc_open_units(df, canonical)
    cfg = load_emission_factor_config(ef_path)
    enriched = apply_proxy_weights(df, cfg)
    return enriched, report


def run(
    csv_path: Path = JRC_UNITS_CSV,
    ef_path: Path = EF_JSON,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched, report = prepare_jrc_units_table(csv_path, ef_path)

    summary_rows = []
    for country, g in enriched.groupby("country", dropna=False):
        for tg, g2 in g.groupby("type_g_normalized", dropna=False):
            summary_rows.append(
                {
                    "country": country,
                    "type_g_normalized": tg,
                    "n_units": len(g2),
                    "capacity_g_mw_sum": float(g2["capacity_g"].sum()),
                    "n_with_coordinates": int((g2["lat"].notna() & g2["lon"].notna()).sum()),
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["country", "type_g_normalized"], na_position="last"
    )

    units_path = output_dir / "jrc_open_units_enriched.csv"
    summary_path = output_dir / "jrc_open_units_by_country_type.csv"
    report_path = output_dir / "jrc_open_units_validation.json"

    enriched.to_csv(units_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return {
        "enriched_csv": units_path,
        "summary_csv": summary_path,
        "validation_json": report_path,
    }


def main() -> None:
    paths = run()
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
