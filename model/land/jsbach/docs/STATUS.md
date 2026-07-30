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

## Open design fork (needs a decision) — snow coefficient build

`calc_snow_abcoeff` (:866-1035) and the newly-formed-layer coefficient re-seeding
(:809-822) use **data-dependent indexing**:

- `t_snow_acoef(ic, itop_old(ic))` — gather a coefficient at a per-cell layer index;
- `grnd_hflx(ic)` / `hcap_grnd(ic)` evaluated at `is = itop(ic)` — scatter/gather at
  a per-cell layer index;
- geometry couples snow to the two uppermost soil layers over `nsnow+2` levels.

This is exactly the gather/scatter question the handoff (`HANDOFF.md` §8.1) flagged
as gating the batching approach. It should be resolved deliberately, not improvised.
Options to weigh: (a) carry the top-layer coefficient as scan state as we cross
`itop` (avoids the gather); (b) precompute an itop-aligned field host-side; (c) use
whatever indexed access current gt4py supports. **Snow is also likely inactive in
the `aes_bubble_land_tmx` validation** (warm desert bubble, 2 h), so this is not on
the slice-1 critical path.

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
