# JSBACH port — status & handoff

Branch `port_jsbach` (off `origin/main`). First vertical slice: soil-snow energy
(SSE). See `sse_port_spec.md` for the verified Fortran requirements and the
icon4py-knowledge design doc (`personal/jcanton/jsbach-port`) for the overall plan.

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
(host-side `zmid`/`zd1` prep, as for soil). **Snow is also likely inactive in the
`aes_bubble_land_tmx` validation** (warm desert bubble, 2 h), so it is not on the
slice-1 critical path regardless.

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
4. **Oracle (M1, long pole)** — instrumentation DONE on the icon-nwp branch
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

## Running the tests

```bash
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend embedded
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend gtfn_cpu
```
