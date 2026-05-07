# C_OtherCombustion

Spatial **allocation** of GNFR **C** CAMS cell emissions onto the reference grid using
Hotmaps + CORINE (**X**), GAINS × Eurostat **end-use buckets** × GAINS **appliance splits** × EMEP (**M**),
then per-cell ``U = X_w @ M.T`` with share normalization and ``np.add.at`` accumulation.

## Entry point

```text
python -m PROXY.main build --sector C_OtherCombustion --country EL
```

- **Sector YAML**: `PROXY/config/sectors/othercombustion.yaml`
- **Science JSON**: `PROXY/config/sectors/other_combustion/` (`GAINS_mapping.json`, `EMEP_emission_factors.json`, `eurostat_end_use.json`)
- **Implementation**: `PROXY/sectors/C_OtherCombustion/` — `pipeline.run` / `run_other_combustion_weight_build`, `m_builder/`, `x_builder/`
- **Entry**: `PROXY.sectors.C_OtherCombustion.pipeline.run_other_combustion_weight_build` (also re-exported from `C_OtherCombustion` package root)

## Pipeline (ASCII)

```text
  GAINS rows ──┐
              ├──► M[p,k] = Σ share × f_enduse[bucket] × f_appliance[class] × EF
  EMEP JSON ──┘              │
                             │    U = X_w · Mᵀ  ──► per-pollutant shares × CAMS E ──► rasters
  Eurostat API (nrg_d_hhq, nrg_bal_s) ──► f_enduse[bucket]
                             ▲
  Hotmaps res / non-res ──► R_base, C_base ──┐
  CORINE L3 ─────────────► mr, mc ──────────┤──► X[pix,k]
  Optional pop / GHSL ────► rural_bias ─────┘
```

## Example (one CAMS cell, toy numbers)

Everything below is **illustrative**; real runs use full rasters, all pollutants, and NetCDF geometry. The **code path** for a single selected GNFR **C** cell `i` is as follows.

**Already computed for the whole domain** (before the cell loop):

- **`X`** — reference grid stack of shape `(H, W, 7)` with bands `R_FIREPLACE` … `C_BOILER_AUT` (Hotmaps + CORINE + optional rural bias).
- **`M`** — matrix of shape `(P, 7)` for that cell’s **ISO3** (`P` = number of output pollutants). Each entry `M[p, k]` comes from GAINS activity × `f_enduse × f_appliance` × EMEP EF for class `k` (see `m_builder/assemble.py`).

**For this one CAMS cell** (`pipeline.py` inner loop):

1. **Bounds** — Read the cell’s lon/lat rectangle from `longitude_bounds` / `latitude_bounds` and intersect the reference grid to get a window `(r0, c0, height, width)`.

2. **Slice proxies** — `Xw = X[r0:r0+h, c0:c0+w, :].reshape(-1, 7)` so each **overlapping pixel** has one 7-vector of relative weights (e.g. 4 pixels ⇒ `Xw` is `4×7`).

3. **Country matrix** — Look up `M` for the cell’s ISO3 (same `M` for every cell in that country).

4. **Per-pixel scores (before normalisation)** — `U = Xw @ M.T` gives a `4×P` array: row = pixel, column = pollutant. Entry `U[j, p]` is proportional to how much of pollutant `p` from this CAMS cell should land on pixel `j`; the allocator then normalises each column to shares that sum to 1 over the window.

5. **Shares and accumulation** (`allocator.accumulate_emissions_and_weights_for_cell`):

   - For each pollutant `p`, let `s = sum_j U[j, p]`. If `s` is positive and finite, **weights** `w_j = U[j, p] / s` (they sum to 1). Otherwise **uniform fallback**: `w_j = 1 / n_pixels`.

   - Read **CAMS emission** `E_p` for this cell from the NetCDF (e.g. `nox` in kg/yr, or CO2 from `co2_ff`/`co2_bf` depending on `co2.mode`).

   - **Scatter-add** onto the national rasters: `acc[p, row_j, col_j] += w_j * E_p` (and optionally the same `w_j` into a weights stack).

**Tiny numeric sketch** — Suppose the window has **one** pixel (`Xw` is `1×7`), two pollutants, and

`Xw = [0.5, 0.3, 0, 0, 0, 0.15, 0.05]` (only fireplace, heating stove, and commercial boilers matter here),

`M = [[1, 2, 0, 0, 0, 0.5, 1], [0, 10, 0, 0, 0, 0, 2]]` (rows = pollutant 0 and 1).

Then `U = Xw @ M.T` is `1×2`:

- `U[0,0] = 0.5·1 + 0.3·2 + 0.15·0.5 + 0.05·1 = 1.225`, `U[0,1] = 0.3·10 + 0.05·2 = 3.1`.

If this cell emits `E_0 = 100` kg/yr for pollutant 0, that **single pixel** receives the full `100` (because it is the only pixel in the window); with more pixels, the same `100` would be split by the normalised `w_j`.

## Logging

Logger name: **`proxy.other_combustion`**. At **INFO**: Eurostat cache/API provenance (per metric where available), `f_enduse_by_bucket` + `f_appliance_by_class` tables (focus country), full **M** for `cams_country_iso3`, **X** band summaries, CAMS mask counts, rural-bias summary, allocator uniform-fallback summary.

## Eurostat

- **Disabled** (`eurostat.enabled: false`, default in `othercombustion.yaml`): all end-use and appliance multipliers are **1.0** (numerically equivalent to the old “Eurostat off” path).
- **Enabled**: fetches `nrg_d_hhq` (household disaggregation) and `nrg_bal_s` (commercial vs household split), caches JSON under `PROXY/cache/eurostat/`. Configure `eurostat.api.timeout_s` and `eurostat.api.offline`.

**Removed**: Table 3 XLSX scraping. Setting `paths.proxy_specific.other_combustion.eurostat_xlsx` raises `ConfigurationError`.

## Rural / urban bias (optional)

YAML block `rural_bias` (see `othercombustion.yaml`). When enabled, uses `paths.population_tif` or `paths.ghsl_smod_tif` (merged from `paths.yaml` in `builder.build`) to scale **R_FIREPLACE** and **R_HEATING_STOVE** bands only.

## Further reading

- `doc/classes.md` — class × proxy × end-use × GAINS × morphology table; why `mr` skips cooking/boilers.
- `doc/corine_mapping.md` — CLC 111/112/121 and weight asymmetry.
- `doc/fallbacks.md` — every fallback and how it is logged.
- `PROXY/docs/other_combustion_refactor_audit.md` — historical audit pointer.
- `PROXY/tests/C_OtherCombustion/` — unit tests for this sector.

## Deprecated

- `run_downscale` alias (unchanged).
- `activity_share_by_class()` diagnostic (emits `DeprecationWarning`; use logged `f_enduse_by_bucket` / `f_appliance_by_class` instead).
