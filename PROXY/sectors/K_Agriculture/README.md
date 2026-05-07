# K_Agriculture (GNFR K/L area proxy)

K_Agriculture builds agriculture area-source weights from a two-stage workflow:

1. A tabular NUTS2 x CORINE model computes pollutant weights from agriculture source
   relevance modules.
2. The raster stage maps those weights to the fine reference grid and normalizes them
   within CAMS GNFR K/L area-source rows.

## Entry Point

`PROXY.main` imports `PROXY.sectors.K_Agriculture.builder` and calls:

```text
build(path_cfg=..., sector_cfg=..., country=...)
```

Typical run:

```text
python -m PROXY.main build --sector K_Agriculture --country EL
```

- **Sector config**: `PROXY/config/sectors/agriculture.yaml`.
- **Alpha config**: `PROXY/config/agriculture/alpha.config.json`.
- **Defaults**: `PROXY/config/agriculture/defaults.json`.
- **Output**: `output_dir` + `output_filename`, one GeoTIFF band per configured pollutant.
- **Tabular side outputs**: `output_dir/tabular/`.

## Folder Boundaries

| Layer | Location | Role |
|-------|----------|------|
| Builder | `builder.py` | Public sector entrypoint, logs root/country, merges config, runs pipeline, verifies GeoTIFF. |
| Pipeline | `pipeline.py` | Orchestrates config merge, tabular weights, reference grid, CAMS K/L raster build, and timing logs. |
| Tabular model | `tabular/` | Agriculture-specific NUTS2 x CLC class extent, rho tables, scores, weights, census IO, and emission-factor loaders. |
| Source relevance | `source_relevance/` | Subsector/process methodology for enteric, manure, fertilized land, rice, biomass burning, housing, soils, liming, and urea. |
| Raster stage | `rasterize_kl.py` | Maps NUTS2 x CLC weights to fine pixels, applies CAMS K/L source rows, writes GeoTIFF + manifest. |
| Shared core | `PROXY.core` | Project-root/path resolution, reference grid, CAMS source-index mapping, GeoTIFF/JSON writing. |

The old sector-local `core/` folder was removed to avoid confusion with shared
`PROXY/core`. Agriculture-specific logic now lives under `tabular/` or
`source_relevance/`.

## Processing Summary

1. Resolve CAMS, CORINE, NUTS, C21, LUCAS, LUCAS soil, GFED, and output paths.
2. Build NUTS2 x agricultural CORINE class extents.
3. Build source-relevance rho tables for processes referenced by `alpha.config.json`.
4. Compute pollutant scores and normalize to NUTS-local CLC weights.
5. Build a CORINE/NUTS reference grid for the selected country.
6. Rasterize NUTS2 ids and read the CORINE class window.
7. Map fine pixels to CAMS source-row indices for GNFR K/L area sources.
8. Assign NUTS2 x CLC pollutant weights to pixels and normalize within each CAMS source row.
9. Write the multiband GeoTIFF and JSON manifest.

## Performance Notes

The pipeline logs timings for major stages: config merge, tabular weights, NUTS2 load,
class extent, each rho process, CSV writes, reference grid, and raster weights.

The main exact-output optimizations are:

- CAMS K/L source-row mapping now uses the shared chunked core mapper instead of building
  full-grid coordinate arrays in the sector.
- Rasterization precomputes pollutant lookup matrices and assigns valid agriculture
  pixels with vectorized indexing instead of looping through CLC classes.
- C21 census counts and GFED arrays are cached on the run config when multiple processes
  need them in one pipeline execution.
- LUCAS prepared points remain cached on the run config as before.

Known expensive stages are NUTS2 x CORINE class histograms, LUCAS/CORINE point sampling,
GFED array loading, and CAMS source-row mapping. The timing logs are the first place to
look when a country run is slow.
