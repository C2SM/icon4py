# JSBACH port — status & handoff

Branch `port_jsbach` (off `origin/main`). First vertical slice: soil-snow energy
(SSE). See `sse_port_spec.md` for the verified Fortran requirements and the
icon4py-knowledge design doc (`personal/jcanton/jsbach-port`) for the overall plan.

> **The oracle loop is closed for slice 1.** ICON has been built and run on Santis and
> the three soil kernels reproduce its savepoints bit-for-bit on `gtfn_cpu` — see
> *Validated against ICON*. Before starting slice 2, read *Oracle gaps*: the dataset is
> single-rank, excludes snow columns, and runs `jsbach_lite`, which constructs only
> about a third of JSBACH's processes.
>
> `SANTIS_BUILD_NOTES.md` in the icon-exclaim.porting working tree records how the
> build, the run and the synthetic land input were done. The dataset is produced by
> `scripts/python/run_serialization.py` (the standard campaign, not a bespoke script)
> from `exp.exclaim_aesPhys_sb` on the ICON branch `serialize_jsbach_sse`.

## Done

Package `icon4py.model.land.jsbach` under a new `model/land/` tree (mirrors the
microphysics package; registered in tach + root workspace/pyproject).

Ported kernels (all GT4Py, TDD'd against numpy transcriptions of the Fortran
recurrences, passing on `embedded` and `gtfn_cpu`):

| kernel                               | Fortran (`mo_sse_process.f90`) | notes                                                                                     |
| ------------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------- |
| `soil_temperature_back_substitution` | calc_soil_temperature :487-504 | forward KDim scan                                                                         |
| `soil_temperature_coefficients`      | calc_soil_temperature :704-743 | reverse scan; bottom division vs interior reciprocal forms kept distinct for bit-fidelity |
| `soil_ground_heat_flux`              | calc_soil_temperature :748-751 | per-level; caller restricts to ground level                                               |
| `snow_temperature_back_substitution` | calc_snow_temperature :796-838 | `itop` mask via per-column index compare in the scan                                      |

Together the three soil kernels are the complete **non-freezing** soil temperature
solve.

Setup / wiring (host-side, `soil_thermal_properties.py`, plain-pytest unit tests):

- `soil_thermal_grid`: soil_depth_energy vertical geometry (dz, mids, bots, zd1)
  from the layer bottom depths (`soillev`), per mo_sse_config_class.f90:238-256.
- `fao_soil_thermal_properties`: per-cell vol_heat_cap / heat_cond from the FAO
  soil-type index (static FAO path, the bubble-validation config), broadcast to
  layers. The FAO lookup is a host-side gather (gather policy). The dynamic
  moisture path (calc_vol_heat_capacity / calc_thermal_conductivity) is the config
  default but is NOT used by the bubble validation, so it is not ported.

Convention: field/argument names follow the JSBACH source (e.g. `t_soil_sl`,
`t_soil_acoef`, `t_srf`), matching the dycore/muphys convention for
Fortran-validated ports; function names are descriptive; docstrings cross-reference
the Fortran `file:line`.

## Validated against ICON

`model/land/jsbach/tests/jsbach/integration_tests/test_sse_datatest.py` replays one
soil-energy step from ICON's savepoints — back substitution with the OLD coefficients,
forward elimination on the new temperatures, then the ground heat flux — and compares
against the exit state, on both `embedded` and `gtfn_cpu`.

**Scope of the claim — the comparison is restricted to snow-free columns.** On
snow-covered columns the top boundary condition and the returned surface quantities are
snow/soil blends the ported kernels do not form, so the test masks them out
(`snow_depth_sl.max(axis=1) == 0`). `snow_temperature_back_substitution` therefore has
**no savepoint evidence behind it** — it is TDD'd against a numpy transcription only.
Likewise only the static FAO thermal-property path is validated; see *Oracle gaps*.

The dataset is `Experiments.EXCLAIM_APE_AES` at **version 11**, grid
`Grids.R02B04_GLOBAL_MPIM` (`icon_grid_0049_R02B04_G`), from `exp.exclaim_aesPhys_sb`
on the ICON branch `serialize_jsbach_sse`. The experiment starts from a **real initial
condition** (`ltestcase = .FALSE.`, `initicon_nml` `init_mode = 2`, the IFS analysis
`ifs2icon_1979010100_0049_R02B04_G.nc`), which is why the dates are 1979-01-01 rather
than the 2008-09-01 of earlier versions.

### Why it is no longer an idealized testcase

v10 used `nh_test_name = 'exclaim_aesPhys'` -- an aqua-planet initial condition with a
land fraction read from `bc_land_frac.nc` -- on synthetic land input. That combination
is not stable once the land input is real, and the reason is structural rather than a
bug: an aqua-planet has an analytic, topography-free atmosphere, while real land data
carries Earth's surface (roughness to 9.8 m against the synthetic 1.5, surface
temperatures 210-313 K against 241-302, lakes or glaciers on 4499 cells). Flat
atmosphere over mountainous surface produces an inconsistent momentum flux from the
first physics call: `vn` at the bottom model level went 35 -> 70 -> 2.7e3 -> 4.5e5 m/s
within one step, and halving the timestep only slowed the divergence (surface `vn`
climbing steadily through 87 m/s) rather than curing it -- which is worse, because the
campaign then "succeeds" and yields a blown-up reference.

The synthetic land worked only because it was bland by construction. That was luck, not
design.

With a real IC the run is clean: three steps, zero CFL warnings, `MAXABS vn = 94 m/s at
level 12` -- the January jet, at jet altitude, not at the surface.

`ltestcase = .FALSE.` also means `init_aes_phy_external` now runs, which is the code
path that properly reads `lsmask`, `glac` and `lake`. The hand-written `CASE` block in
`mo_aes_phy_init.f90` and the `mo_nh_testcases.f90` entries are no longer exercised by
this experiment; they are retained but are dead for it.

### Input data, and a trap worth remembering

Land input is the MPI-M published data, `0049/land/r0103` -- not synthetic, and not the
older revisions:

- **grid 0013 cannot be used.** Its last land revision (r0003, 2019) predates fields the
  current JSBACH requires: the run aborts on `residual_water not found`. Grid 0049 is the
  R02B04 that carries a current (r01xx) revision.
- **r0103 over r0101** because r0101 lacks `skin_conductivity`, which SEB reads behind a
  `Has_var` guard -- absent, it silently takes a fallback instead of the production path,
  and a silently-wrong oracle is worse than a crash. r0103 also derives soil depth from
  the extpar root depth rather than deprecated tuning, which sets the SSE vertical grid.
- **SST/sea-ice is required** whenever tmx or radiation is on
  (`mo_aes_phy_init.f90:719`, `dt_rad > 0 .OR. dt_vdf > 0`), with no `ltestcase` escape.
  Taken from `sst_and_seaice/r0100` -- **not** the `r0001` the AMIP reference experiments
  link, which is on a different mesh.

**Always check `uuidOfHGrid` on every input file against the grid before submitting.**
Two separate mistakes in this series -- 0013 land data, and the r0001 SST -- were files
of the right name and right cell count on the wrong mesh. The check takes seconds; each
miss cost a campaign.

Archive size was held down by gating the 11 per-substep dycore savepoints behind a new
`dycore_internals` namelist switch (default `.TRUE.`, off in this experiment) plus a
shorter serialization window: 362 -> 66 savepoints per rank. Reducing `ndyn_substeps`
was considered and rejected -- it rescales the diffusion coefficients and the divergence
damping, i.e. it changes the reference solution.

**The gate is not bit-exactness, and that is a finding rather than a compromise.** On
`gtfn_cpu` all five compared quantities are bit-identical to ICON; on `embedded` they
sit a few ulp away, because ICON (nvfortran) and gtfn (g++) both contract `a + b*c` into
a fused multiply-add and numpy does not. For `grnd_hflx` that single unfused operation
shows up as ~1e-10 relative, since the expression cancels two O(300 K) terms down to
O(1 W/m²). See `fma_contraction.md` — it is written up for @muellc, because the
consequence ("bit-exact against ICON" is a property of the backend, not of the port)
reaches every Fortran-validated port, not just this one.

## Snow coefficient build — resolved approach (no new GT4Py capability needed)

`calc_snow_abcoeff` (:866-1035) and the newly-formed-layer coefficient re-seeding
(:809-822) contain the only two **data-dependent gathers** in the snow path:

1. re-seed `t_snow_acoef/bcoef(ic, itop_old(ic))` (:817-818) — read a coefficient at
   a per-cell layer index;
2. `grnd_hflx(ic)` / `hcap_grnd(ic)` evaluated at `is = itop(ic)` (:1026-1028).

GT4Py has **no absolute field indexing** — only relative K-offsets (`field(Koff[1])`)
— so neither is expressible as a stencil access. (The `is>itop` / `is>=itop` masks are
NOT gathers, just per-level comparisons, and are already handled, e.g. in
`snow_temperature_back_substitution`.)

**Project decision (policy): handle every such per-cell gather host-side (option b),
always.** The granule/orchestration layer gathers with `array_ns` (backend-agnostic:
numpy on CPU, cupy on GPU — no host round-trip), producing a plain `CellField` the
stencils consume; stencils stay index-free. This is the port's anticorruption layer
against JSBACH's per-cell index idioms, has direct icon4py precedent
(`compute_diffusion_metrics.py:186` uses `array_ns.take_along_axis`), and generalises
to the many gathers still to come (HYDRO etc.). All physics *arithmetic* stays in
GT4Py; only data movement/selection is host-side, so bit-reproducibility is unaffected.

Applied to the two snow gathers:

- **(1) re-seed** `seed[ic] = t_snow_acoef[ic, itop_old[ic]]` via `array_ns.take_along_axis`
  → `CellField`, then a masked stencil fills layers `k in [itop, itop_old)`.
- **(2) grnd_hflx/hcap_grnd @itop**: a stencil computes the flux per level as a
  `CellKField`; the host selects the `itop` level with `array_ns.take_along_axis`.

Future migration: if/when GT4Py grows a native gather, replace these host ops with it
so the DSL is not left and re-entered — the seam is small and localised by design.

Geometry note: snow couples to the two uppermost soil layers over `nsnow+2` levels
(host-side `zmid`/`zd1` prep, as for soil). The v10 dataset does contain snow-covered
columns, but the datatest excludes them (see *Validated against ICON*), so the snow
path remains unvalidated against ICON.

⚠️ Verify once empirically before relying on this in a GPU hot path: that interleaving
a per-timestep `array_ns` gather with GT4Py programs does not force a stream sync /
block whole-granule graph capture. (icon4py's existing `array_ns` gathers are all
one-time setup, not per-step.)

## Not yet done (next steps)

1. **SSE orchestration (granule)** — a `model.Component`-style process module (the
   icon4py `setup_program` pattern) assembling back-sub → coefficients → ground flux
   in the calc_soil_temperature order, wired to the geometry/properties above. Best
   done alongside the oracle so the assembled step can be validated end-to-end, not
   just by numpy self-consistency.
2. **Freeze/melt + thaw depth** (:507-687) — the bubble config sets `l_freeze=.TRUE.`
   but on warm desert it is likely a no-op (soil > tmelt); confirm against the oracle,
   then port if it fires. `l_supercool=.FALSE.`.
3. **Dynamic thermal properties** — `calc_vol_heat_capacity` /
   `calc_thermal_conductivity` (the `l_heat_cap_dyn`/`l_heat_cond_dyn` default path,
   moisture-coupled to HYDRO). Needed for non-bubble experiments; not on the bubble
   validation path.
4. ~~**Oracle (M1, long pole)**~~ — **DONE for slice 1**, see *Validated against ICON*
   and *Oracle gaps* below. It is deliberately built to grow: one experiment, one
   dataset version, savepoints added per slice.
   Historical note: instrumentation on the icon-nwp branch
   `serialize_jsbach_sse` (off `serialize_tmx_sfc`): `serialize_sse_entry/exit/geometry`
   in `mo_icon4py_verification.f90`, call sites in `update_land`, and the experiment
   `exp.aes_bubble_land_tmx_sse_ser` (`l_freeze=.FALSE.`, serialization on). See that
   branch's `JSBACH_SSE_VALIDATION.md`. Remaining: build + run ICON with the land pool
   data to emit the savepoints (Serialbox2 builds and ICON configures on this mac, but
   the run needs the land input files, so generation is on the ICON machine), then
   register the dataset in icon4py `definitions.py` and add the datatest that drives
   the three kernels against `sse-solve-exit`. Grid decision still open (below).
5. **tmx seam** — replace the prescribed `land_*` fields in the `tmx-surface`
   worktree once that stabilises.

## Oracle gaps (read before starting slice 2)

The oracle is one experiment (`exclaim_aesPhys`) carrying one dataset version, and it is
meant to grow: each new slice adds savepoints to `mo_icon4py_verification.f90`, bumps
the version, and reuses the grid, the synthetic land input and the campaign unchanged.
What is *not* yet general:

| gap | effect | cost to close |
| --- | --- | --- |
| `comm_size = 1` only | no MPI/distributed tests | rerun campaign for 2 and 4 |
| snow columns excluded | snow path unvalidated | needs the snow/soil blend ported, not new data |
| `l_freeze = .FALSE.` | freeze/melt + thaw depth unvalidated | namelist flip + new savepoints |
| `l_heat_cap/cond_dyn = .FALSE.` | moisture-coupled thermal properties unvalidated | coupled to HYDRO; flip once HYDRO is ported |
| `usecase = 'jsbach_lite'` | only ~8 of 25 processes ever run | see below — the structural one |

**The structural gap is the usecase.** `jsbach_lite` + tmx instantiates only
`A2L_, L2A_, SEB_, RAD_, HYDRO_, TURB_, SSE_, PHENO_` (+ `HD_` if active) —
`mo_jsb_model_usecases.f90:init_usecase_lite_tmx`. The other sixteen process
directories (carbon, assimilation, disturbance, natural/anthropogenic LCC, forest age,
the `q_*` QUINCY set) are never constructed, so no namelist or input file will make
them emit savepoints. Reaching them means `jsbach_pfts` or `quincy_13_pfts`, which
bring a different tile structure (PFTs below the vegetation tile), an `lctlib` table,
and substantially more boundary data than the 26 synthetic variables now generated.

That is a decision to take deliberately rather than by drift: growing the *same*
experiment through `lite → pfts → quincy` keeps one oracle, but each step changes the
tile tree and therefore the land input generator.

## Slice sizing (`jsbach_lite`, the reachable set)

Fortran line counts, whole process directory / `*_process.f90` core:

| process | dir | core | notes |
| --- | --- | --- | --- |
| SSE | 3 875 | 1 069 | slice 1 — partially done |
| TURB | 1 837 | — | small; feeds the tmx seam |
| PHENO | 3 158 | 999 | |
| SEB | 4 494 | — | surface energy balance |
| RAD | 5 286 | 1 761 | |
| HYDRO | 11 833 | 3 526 | largest; SSE's dynamic properties depend on it |
| A2L/L2A | 2 505 | — | coupling layer, not physics |
| HD | 2 229 | — | only if `active` |

## Running the tests

```bash
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend embedded
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend gtfn_cpu
```
