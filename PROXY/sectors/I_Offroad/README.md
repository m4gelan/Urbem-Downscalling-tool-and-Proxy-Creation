# I_Offroad (GNFR I area proxy)

I_Offroad builds multiband area-source weights for three offroad legs:
`rail` (`1A3c`), `pipeline` (`1A3ei`), and `nonroad` (`1A3eii`). For each pollutant
band, CEIP country shares mix the three proxy rasters before the result is normalized
within CAMS cells.

## Entry Point

`PROXY.main` imports `PROXY.sectors.I_Offroad.builder` and calls:

```text
build(path_cfg=..., sector_cfg=..., country=...)
```

Typical run:

```text
python -m PROXY.main build --sector I_Offroad --country EL
```

- **Sector config**: `PROXY/config/sectors/offroad.yaml`.
- **Output**: `output_dir` + `output_filename`, one GeoTIFF band per configured pollutant.
- **Manifest**: JSON sidecar beside the GeoTIFF.

## Folder Boundaries

| Layer | Location | Role |
|-------|----------|------|
| Builder | `builder.py` | Public sector entrypoint, config merge, input/output logging, pipeline call, output verification. |
| Pipeline | `pipeline.py` | I-specific orchestration: reference grid, CEIP triple shares, country raster, CAMS cells, proxy mix, normalization, write. |
| Rail proxy | `rail_osm.py` | OSM rail lines plus `landuse_railway`, buffered coverage, z-score. |
| Pipeline proxy | `pipeline_osm.py` | OSM pipeline lines/areas plus optional facilities raster blend. |
| Nonroad proxy | `nonroad_corine_only.py` | CORINE agriculture and industrial masks only. |
| Core helpers | `PROXY.core.alpha`, `PROXY.core.corine.raster`, `PROXY.core.grid`, `PROXY.core.raster` | CEIP/YAML fallback shares, CORINE nearest warp, reference grid, country/CAMS raster helpers, output writing. |

`multiband_builder.py` was removed; its active build flow now lives in `pipeline.py`.
`offroad_area_weights.py` and `cams_area_mask.py` remain for visualization helpers.

## Processing Summary

1. Build the reference window from CORINE and NUTS.
2. Warp CORINE classes to the reference grid.
3. Build three raw proxies: rail OSM, pipeline OSM/facilities, and nonroad CORINE.
4. Read CEIP shares for `(rail, pipeline, nonroad)` per country and pollutant.
5. Apply YAML alpha fallbacks only where CEIP is missing or equal to the default triple.
6. Rasterize NUTS countries so each fine pixel uses the correct CEIP country shares.
7. Build the CAMS cell ID grid and normalize each pollutant proxy within each CAMS cell.
8. Write the multiband GeoTIFF and JSON manifest.

## Configuration Notes

- `pollutants` defines output band order after pollutant-key normalization.
- `defaults.default_shares_*` defines the fallback triple when CEIP has no usable row.
- `cntr_code_to_iso3` maps NUTS country tokens to the ISO3 keys used by CEIP.
- `proxy.rail_buffer_m`, `proxy.pipeline_buffer_m`, and `proxy.osm_subdivide` control OSM coverage.
- `proxy.w_agri` and `proxy.w_ind` mix the nonroad CORINE agriculture and industrial channels.
- `facilities_tif` is optional; when absent, the pipeline proxy uses only OSM.
