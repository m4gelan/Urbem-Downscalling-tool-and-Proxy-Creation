# D_Fugitive (GNFR D area proxy)

Fugitive emissions are split into **CEIP groups** (`G1`–`G4`), each with OSM + CORINE
rules in `config/ceip/profiles/fugitive_groups.yaml`. For each CAMS cell and each
**pollutant** band, the pipeline combines **CEIP-reported country–group–pollutant
weights (α)** with **spatial proxies** (OSM, CORINE, population) built on the same
reference grid as CORINE + NUTS.

This sector uses the **same shared implementation** as `B_Industry`; only config keys,
default CEIP profile, OSM path alias, and logging tag differ.

## Entry point

`PROXY.main` imports `PROXY.sectors.D_Fugitive.builder` and calls `build(path_cfg=…, sector_cfg=…, country=…)`:

```text
python -m PROXY.main build --sector D_Fugitive --country EL
```

- **Sector config**: `PROXY/config/sectors/fugitive.yaml` (listed in `PROXY/config/sectors.yaml`).
- **Output**: multiband GeoTIFF path from `output_dir` + `output_filename` (one band per pollutant in `pollutants`).

## How this folder relates to shared code

| Layer | Location | Role |
|-------|----------|------|
| **Builder** | `builder.py` | Resolves paths, builds the **reference window** (`reference_window_profile`), merges YAML into the structure expected by the pipeline (`merge_ceip_group_sector_cfg` in `PROXY.sectors._shared.gnfr_groups`), calls `run_fugitive_pipeline`. |
| **Thin wrapper** | `pipeline.py` | `run_fugitive_pipeline` → `run_gnfr_group_pipeline` with `sector_key="D_Fugitive"` and default OSM loader (industry passes a multi-layer GPKG reader). |
| **Core algorithm** | `PROXY.sectors._shared.gnfr_groups` | `run_gnfr_group_pipeline`: CAMS ids, rasters, α from CEIP, per-group and per-pollutant normalization, multiband write. |
| **CEIP / α** | `PROXY.core.alpha.ceip.reported_group_alpha` | `load_ceip_and_alpha` (GNFR group sectors; workbook paths come from merged config). Re-exported from `PROXY.core.alpha.ceip`. |

Methodology details: `PROXY/docs/sector_methodology.md` (GNFR D / fugitive section, if present).

## Configuration notes

- **`pollutants`**: list order defines **output band order**; must align with the α tensor built from the CEIP workbook. If omitted, the shared default is `DEFAULT_CEIP_GROUP_POLLUTANTS` in `gnfr_groups.py`.
- **`cams_country_iso3`**: used as fallback when mapping NUTS pixels to CEIP **country** rows (see pipeline logging / `run_gnfr_group_pipeline`).
- **`fugitive_paths`**: CEIP workbook, groups YAML, optional `cntr_code_to_iso3` overrides (see `fugitive.yaml`).
- **`ceip_group_order`** / **`group_order`** / nested **`ceip.group_order`**: ordered list of keys under **`groups:`** in the CEIP profile YAML. Default is `G1`–`G4`.
