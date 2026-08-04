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

1. **Geometry prep** — `dz`, `zd1 = 1/(mids(k+1)-mids(k))` are currently kernel
   inputs; wire them from the `soil_depth_energy` vgrid (one-time, host-side).
2. **Soil properties** — `vol_heat_cap`, `heat_cond` come from the SSE properties
   task (fao index → parameter tables); port `calc_*` for those or prescribe from
   serialized data.
3. **Freeze/melt + thaw depth** (:507-687) — deferred; start with
   `l_freeze=.FALSE.`, `l_supercool=.FALSE.`.
4. **SSE orchestration** — a process module assembling back-sub → coefficients →
   ground flux in the calc_soil_temperature order, wired to inputs.
5. **Oracle (M1, long pole)** — needs the Fortran machine: add JSBACH savepoints to
   `exp.aes_bubble_land_tmx` (tier-2) and/or an offline single-column standalone run
   (tier-1); register the 20x4 torus grid + experiment in icon4py.
6. **tmx seam** — replace the prescribed `land_*` fields in the `tmx-surface`
   worktree once that stabilises.

## Running the tests

```bash
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend embedded
uv run --group test --frozen pytest model/land/jsbach/tests/ --backend gtfn_cpu
```
