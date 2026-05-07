# G_Shipping (GNFR G area proxy)

Shipping uses one shared spatial proxy for all GNFR G pollutants. There is no CEIP
subsector split and no per-pollutant alpha matrix at this stage: the output is a
single CAMS-cell-normalized weight band.

## Entry Point

`PROXY.main` imports `PROXY.sectors.G_Shipping.builder` and calls:

```text
build(path_cfg=..., sector_cfg=..., country=...)
```

Typical run:

```text
python -m PROXY.main build --sector G_Shipping --country EL
```

- **Sector config**: `PROXY/config/sectors/shipping.yaml`.
- **Defaults**: `PROXY/config/shipping/defaults.json`.
- **Output**: `output_dir` + `output_filename`, defaulting to `shipping_areasource.tif`.

## How This Folder Relates To Shared Code

| Layer | Location | Role |
|-------|----------|------|
| **Builder** | `builder.py` | Entrypoint used by `PROXY.main`; delegates path/config merge and runs the pipeline. |
| **Pipeline** | `pipeline.py` | Resolves CAMS, CORINE, NUTS, EMODnet, and OSM paths; builds the reference profile with `PROXY.core.ref_profile.load_area_ref_profile`; calls `run_shipping_areasource`. |
| **Proxy logic** | `proxy_shipping.py` | Builds the shipping proxy from EMODnet vessel density, OSM shipping layers, and CORINE port coverage; normalizes it within CAMS cells and writes the GeoTIFF. |
| **Core helpers** | `PROXY.core` | CAMS cell ids (`core.cams.grid.build_cam_cell_id`), OSM/CORINE coverage (`core.osm_corine_proxy`), and within-CAMS normalization / validation (`core.raster`). |

## Proxy Construction

The raw shipping proxy is:

```text
P = w_emodnet * D_n_damped + w_osm * z(OSM) + w_port * z(CLC port)
```

- `D_n_damped`: EMODnet vessel density warped to the reference grid and damped on land.
- `z(OSM)`: normalized OSM shipping coverage from `osm_shipping_high` and `osm_shipping_medium`.
- `z(CLC port)`: normalized CORINE port-area coverage, default CLC level-2 code `123`.

The weights come from `shipping.proxy` in config and are renormalized internally if their
sum is not exactly 1.

## CAMS Normalization

`proxy_shipping.py` maps each fine pixel to a CAMS geographic cell with
`build_cam_cell_id`, then calls `normalize_within_cams_cells` so the weight band sums to
1 inside each CAMS cell. If a CAMS cell has zero proxy mass, the shared core helper uses
a uniform fallback over that cell and logs a summary.

Pixels outside CAMS get nodata `-9999`.

## Diagnostics

Set `shipping.proxy.write_diagnostics: true` or `shipping.output.write_diagnostics: true`
to write diagnostic rasters:

- `emodnet_raw`
- `D_n_damped`
- `osm_coverage`
- `clc_port_frac`
- `z_osm`
- `z_clc`
