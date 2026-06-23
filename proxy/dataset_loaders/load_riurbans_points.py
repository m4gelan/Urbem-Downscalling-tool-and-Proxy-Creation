from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from proxy.core import log

# RI-URBANS CSV columns -> proxy pollutant labels (point sources only; SourceType P is fixed in loader).
RIURBANS_POLLUTANT_MAP = {
    "CO": "CO",
    "NOX": "NOx",
    "SO2": "SOx",
    "NMVOC": "NMVOC",
    "NH3": "NH3",
    "PM10": "PM10",
    "PM2_5": "PM2.5",
}


def load_riurbans_points(
    csv_path: Path,
    *,
    country_iso3: str,
    gnfr_sectors: list[str],
) -> dict[str, dict[str, Any]]:
    iso3 = str(country_iso3).strip().upper()
    gnfr_set = {str(g).strip() for g in gnfr_sectors if str(g).strip()}
    if not gnfr_set:
        raise ValueError("gnfr_sectors must be non-empty")

    out: dict[str, dict[str, Any]] = {}
    row_i = 0
    for chunk in pd.read_csv(csv_path, sep=";", decimal=".", chunksize=500_000):
        df = chunk.loc[
            (chunk["ISO3"].astype(str).str.upper() == iso3)
            & (chunk["SourceType"].astype(str).str.upper() == "P")
            & (chunk["GNFR_Sector"].astype(str).str.strip().isin(gnfr_set))
        ]
        for r in df.itertuples(index=False):
            d = r._asdict()
            pollutants: dict[str, float] = {}
            for ri_col, label in RIURBANS_POLLUTANT_MAP.items():
                val = float(d.get(ri_col) or 0.0)
                if val > 0.0:
                    pollutants[label] = val
            pid = f"ri_{row_i}"
            row_i += 1
            out[pid] = {
                "lon": float(d["Lon"]),
                "lat": float(d["Lat"]),
                "gnfr": str(d["GNFR_Sector"]).strip(),
                "year": int(d["Year"]),
                "pollutants": pollutants,
                "facility_id": pid,
            }

    log.info(
        f"RI-URBANS point sources: {len(out)} for {iso3} GNFR {sorted(gnfr_set)} "
        f"from {csv_path.name}"
    )
    return out
