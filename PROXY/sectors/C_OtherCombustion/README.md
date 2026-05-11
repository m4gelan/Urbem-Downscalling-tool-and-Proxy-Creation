# C_OtherCombustion

Spatial **allocation** of GNFR **C** CAMS cell emissions onto the reference grid using
Hotmaps + CORINE (**X**), GAINS × Eurostat **end-use buckets** × GAINS **appliance splits** × EMEP (**M**),
then per-cell share normalisation and ``np.add.at`` accumulation.

When ``run.enable_offroad`` is true (default), each pollutant’s weights combine a **stationary**
branch (column normalisation of ``U = X_w @ M.T``) with an **off-road** branch: three
spatial proxies (forestry CLC, residential CLC112 + population blend, commercial CLC121 + OSM)
weighted by CEIP reported-emissions α on groups **G1**–**G4** in
``PROXY/config/ceip/profiles/C_OtherCombustion_groups.yaml``, with tunables in
``C_OtherCombustion_rules.yaml``. Use ``python -m PROXY.main build --sector C_OtherCombustion --no-enable-offroad``
for legacy stationary-only behaviour.

**Stationary X (seven bands):** with ``appliance_proxy.enabled: true`` in ``othercombustion.yaml`` (default), ``X_k = S_k × L_k`` uses warped **POP**, **H_res / H_nres**, **HDD** (Hotmaps ``hdd_curr``), **GHS-SMOD** (rural settlement mask vs CLC 111/112/121), and **CLC L3** indicators only — configured under ``appliance_proxy`` in ``C_OtherCombustion_rules.yaml``. Set ``appliance_proxy.enabled: false`` to restore the legacy stack (**Hotmaps heat + GFA** ``R_base`` / ``C_base``, CORINE ``mr`` / ``mc``, optional ``rural_bias``).

## GNFR C: stationary + off-road branches

| Subgroup | NFR (CEIP) | Spatial proxy |
|----------|------------|----------------|
| Stationary | ``1A4ai``, ``1A4bi``, ``1A4ci`` (G1) | Default: S×L appliance ``X`` (rules YAML); legacy: Hotmaps×GFA + CORINE ``mr``/``mc`` → ``U`` |
| Forestry off-road | ``1A4cii`` (G2) | CLC forest classes (default 311–313), uniform fallback |
| Residential off-road | ``1A4bii`` (G3) | CLC 112 + min–max population blend (shared with PublicPower formula) |
| Commercial off-road | ``1A4aii`` (G4) | CLC 121 + λ × OSM (``osm_commercial`` rules in rules YAML) |

Per pollutant: ``α_stat = α_G1``, ``α_off = α_G2+α_G3+α_G4``, ``β_F = α_G2/α_off``, etc. (EU pool from ``alpha_methods.yaml``). Optional per-pollutant overrides in ``alpha_beta_override`` in ``C_OtherCombustion_rules.yaml``.

Constants: ``PROXY/sectors/C_OtherCombustion/nfr_codes.py`` (`NFR_STATIONARY`, `NFR_OFFROAD`, `NFR_ALL_C`).

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
  Appliance mode (default): POP, H_res/H_nres, HDD, GHS-SMOD, CLC L3 ──► S_k, L_k ──► X[pix,k]
  Legacy mode: Hotmaps res/non-res heat+GFA ──► R_base, C_base ──┐
  CORINE L3 (legacy) ──► mr, mc ────────────────────────────────┤──► X[pix,k]
  Optional pop / GHSL (legacy rural_bias only) ────────────────┘
```

## Example (one CAMS cell, toy numbers)

Everything below is **illustrative**; real runs use full rasters, all pollutants, and NetCDF geometry. The **code path** for a single selected GNFR **C** cell `i` is as follows.

**Already computed for the whole domain** (before the cell loop):

- **`X`** — reference grid stack of shape `(H, W, 7)` with bands `R_FIREPLACE` … `C_BOILER_AUT` (default: **S×L appliance proxy** from rules YAML; or legacy Hotmaps + CORINE + optional rural bias).
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

## Rural / urban bias (legacy X only)

YAML block `rural_bias` (see `othercombustion.yaml`). When **legacy** Hotmaps×GFA X is enabled (`appliance_proxy.enabled: false`), uses `paths.population_tif` or `paths.ghsl_smod_tif` to scale **R_FIREPLACE** and **R_HEATING_STOVE** bands only. The default **appliance** X already encodes a GHS×CLC rural term in `S_k`; do not enable `rural_bias` alongside appliance mode to avoid double structure.

## Further reading

- `doc/classes.md` — class × proxy × end-use × GAINS × morphology table; why `mr` skips cooking/boilers.
- `doc/corine_mapping.md` — CLC 111/112/121 and weight asymmetry.
- `doc/fallbacks.md` — every fallback and how it is logged.
- `PROXY/docs/other_combustion_refactor_audit.md` — historical audit pointer.
- `PROXY/tests/C_OtherCombustion/` — unit tests for this sector.

## Deprecated

- `run_downscale` alias (unchanged).
- `activity_share_by_class()` diagnostic (emits `DeprecationWarning`; use logged `f_enduse_by_bucket` / `f_appliance_by_class` instead).
