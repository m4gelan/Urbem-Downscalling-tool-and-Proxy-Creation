# B_Industry (GNFR B area proxy)

Industry emissions are split into **CEIP groups** (`G1`–`G4`), each with OSM + CORINE
rules in `config/ceip/profiles/industry_groups.yaml`. For each CAMS cell and each
**pollutant** band, the pipeline combines **CEIP-reported country–group–pollutant
weights (α)** with **spatial proxies** (OSM, CORINE, population) built on the same
reference grid as CORINE + NUTS.

## Entry point

`PROXY.main` imports `PROXY.sectors.B_Industry.builder` and calls `build(path_cfg=…, sector_cfg=…, country=…)`:

```text
python -m PROXY.main build --sector B_Industry --country EL
```

- **Sector config**: `PROXY/config/sectors/industry.yaml` (listed in `PROXY/config/sectors.yaml`).
- **Output**: multiband GeoTIFF path from `output_dir` + `output_filename` (one band per pollutant in `pollutants`).

## How this folder relates to shared code

| Layer | Location | Role |
|-------|----------|------|
| **Builder** | `builder.py` | Resolves paths, builds the **reference window** (`reference_window_profile`), merges YAML into the structure expected by the pipeline (`merge_ceip_group_sector_cfg` in `PROXY.sectors._shared.gnfr_groups`), calls `run_industry_pipeline`. |
| **Thin wrapper** | `pipeline.py` | `run_industry_pipeline` → `run_gnfr_group_pipeline` with `sector_key="B_Industry"`, industry OSM loader (all GPKG layers). |
| **Core algorithm** | `PROXY.sectors._shared.gnfr_groups` | `run_gnfr_group_pipeline`: CAMS ids, rasters, α from CEIP, per-group and per-pollutant normalization, multiband write. |
| **CEIP / α** | `PROXY.core.alpha.ceip.reported_group_alpha` | `load_ceip_and_alpha` (shared GNFR group CEIP α; sector-specific workbook paths come from config). Also `from PROXY.core.alpha.ceip import load_ceip_and_alpha`. |

Methodology details: `PROXY/docs/sector_methodology.md` (B_Industry section).

## Configuration notes

- **`pollutants`**: list order defines **output band order**; must align with the α tensor built from the CEIP workbook. If omitted, the shared default is `DEFAULT_CEIP_GROUP_POLLUTANTS` in `gnfr_groups.py`.
- **`cams_country_iso3`**: used as fallback when mapping NUTS pixels to CEIP **country** rows (see pipeline logging / `run_gnfr_group_pipeline`).
- **`ceip_group_order`** / **`group_order`** / nested **`ceip.group_order`**: ordered list of keys that must exist under **`groups:`** in the CEIP profile YAML (same names as in the workbook’s mapped **group** column). Default is `G1`–`G4`. For another sector reusing this pipeline (e.g. three legs), use semantic ids such as `rail`, `pipeline`, `nonroad` and define matching blocks under `groups:` plus matching keys in `PROXY/config/alpha/fallback/` when you use YAML alpha overrides.

