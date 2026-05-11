# E_Solvents (GNFR E area proxy)

Solvent emissions are split into nine **2.D.3 subsectors** (`d3a_domestic` through
`d3i_other`). For each CAMS GNFR E area-source parent and pollutant band, the pipeline
combines **CEIP-reported country-subsector-pollutant weights (alpha)** with solvent
spatial proxies built from population, CORINE, OSM landuse/buildings/roads, and the
configured beta/archetype weights.

## Entry point

`PROXY.main` imports `PROXY.sectors.E_Solvents.builder` and calls:

```text
build(path_cfg=..., sector_cfg=..., country=...)
```

Typical run:

```text
python -m PROXY.main build --sector E_Solvents --country EL
```

- **Sector config**: `PROXY/config/sectors/solvents.yaml`.
- **Defaults**: `PROXY/config/solvents/defaults.json`.
- **Output**: multiband GeoTIFF path from `output_dir` + `output_filename`, one band per pollutant in `pollutants`.

## How this folder relates to shared code

| Layer | Location | Role |
|-------|----------|------|
| **Builder** | `builder.py` | Resolves paths, merges sector/default config, logs resolved inputs, calls `run_solvents_pipeline`, and verifies the written output path. |
| **Config / public runner** | `pipeline.py` | `merge_solvents_pipeline_cfg` plus the public `run_solvents_pipeline` wrapper. |
| **Pipeline body** | `run_pipeline.py` | E-specific orchestration: CEIP alpha, CAMS source rows, raw indicators, archetypes, subsector beta stack, within-CAMS normalization, allocation, validation, and write. |
| **CEIP / alpha** | `PROXY.core.alpha.ceip` | `load_ceip_and_alpha_solvents` reads the reported-emissions workbook and `PROXY/config/ceip/profiles/solvents_subsectors.yaml`. |
| **Reusable raster math** | `PROXY.core.raster`, `PROXY.core.area_allocation`, `PROXY.core.corine.raster` | Population sum warp, quantile normalization, within-CAMS stack normalization, strict validation, CLC masks, and alpha-weight allocation. |
| **Solvent methodology** | `archetypes.py`, `subsector_proxies.py`, `osm_indicators.py` | Solvent-specific archetype channels, beta mixing, and OSM channel construction. |

## Configuration notes

- **`pollutants`**: list order defines output band order and the alpha matrix row order.
- **`subsectors`**: list order defines the alpha and `rho_raw` stack column order; keys must match `solvents_subsectors.yaml`, `beta`, and fallback alpha config.
- **`ceip_years` / `ceip_year`**: `ceip_year` is converted to a one-year `ceip_years` filter when no explicit `ceip_years` list is set. Omit both to average all reported workbook years.
- **`archetype_weights`**: mixes raw indicators into `house`, `serv`, `ind`, and `infra`.
- **`beta`**: mixes those four archetypes into the nine solvent subsectors; each row must sum to 1.
- **`SOLVENTS_SKIP_OSM=1`**: skips OSM reads and returns zero OSM channels for dry runs.

## Processing summary

1. Build a reference grid from `ref_tif` or CORINE + NUTS.
2. Read CEIP reported emissions and compute `alpha[pollutant, subsector]`.
3. Map fine pixels to CAMS GNFR E area-source rows.
4. Build raw indicators from population, CORINE masks, and OSM solvent channels.
5. Normalize indicators into four archetypes, then apply `beta` to create subsector proxies.
6. Normalize each subsector proxy within each CAMS parent row; fallback is generic proxy, then uniform.
7. Combine normalized subsector proxies with alpha to produce pollutant weights.
8. Validate non-negative values and `sum(pixel weights in each CAMS parent) = 1`.
