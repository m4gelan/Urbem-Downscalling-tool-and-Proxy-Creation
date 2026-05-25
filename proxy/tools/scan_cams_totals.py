from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

_repo = Path(__file__).resolve().parents[3]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from proxy.core.alias import cams_pollutant_var

CONFIG = _repo / "proxy/config/sector/D_Fugitive/D_Fugitive_sector_config.yaml"
SOURCE_AREA = 1
SOURCE_POINT = 2


def _decode_country_id(raw) -> str:
    return str(raw.decode("utf-8") if isinstance(raw, bytes) else raw).strip().upper()


def main() -> None:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cams_path = _repo / str(cfg["filepaths"]["CAMS"]["path"]).replace("\\", "/")
    pollutants = [str(p).strip() for p in cfg["pollutants"]]
    ec_filter = np.asarray(cfg["cams_point_sources"]["emission_category_indices"], dtype=np.int64)

    if not cams_path.is_file():
        raise FileNotFoundError(f"CAMS file not found: {cams_path}")

    with xr.open_dataset(cams_path, engine="netcdf4") as ds:
        country_ids = [_decode_country_id(x) for x in ds["country_id"].values]
        emis_cat = np.asarray(ds["emission_category_index"].values, dtype=np.int64).ravel()
        src_type = np.asarray(ds["source_type_index"].values, dtype=np.int64).ravel()
        country_index = np.asarray(ds["country_index"].values, dtype=np.int64).ravel()

        pol_arrays = {
            lab: np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float64).ravel(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for lab in pollutants
        }

    sector_mask = np.isin(emis_cat, ec_filter)
    active_countries = sorted(
        {
            country_ids[int(ci) - 1]
            for ci in np.unique(country_index[sector_mask])
            if 1 <= int(ci) <= len(country_ids)
            and np.any(
                sector_mask
                & (country_index == int(ci))
                & (np.stack(list(pol_arrays.values()), axis=1).max(axis=1) > 0.0)
            )
        }
    )

    print(f"CAMS file: {cams_path.name}")
    print(f"Sector D_Fugitive (emission_category {ec_filter.tolist()})")
    print(f"Units: kg/year")
    print(f"Countries with non-zero emissions: {len(active_countries)}")
    print()

    for iso3 in active_countries:
        ci = country_ids.index(iso3) + 1
        country_mask = sector_mask & (country_index == ci)
        area_mask = country_mask & (src_type == SOURCE_AREA)
        point_mask = country_mask & (src_type == SOURCE_POINT)

        print(f"=== {iso3} ===")
        print(f"{'pollutant':<8} {'area_kg_yr':>16} {'point_kg_yr':>16} {'total_kg_yr':>16}")
        for lab in pollutants:
            area = float(pol_arrays[lab][area_mask].sum())
            point = float(pol_arrays[lab][point_mask].sum())
            print(f"{lab:<8} {area:16.3f} {point:16.3f} {area + point:16.3f}")
        print()


if __name__ == "__main__":
    main()
