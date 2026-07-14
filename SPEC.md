# Spec: Bitwise Identical GPU Results for Standalone Driver MPI Tests

## Current state

- Halo exchange fix for `z_nabla2_e` applied (commit `ab078cdb9c`)
- GPU flags + zero tolerance + print-not-fail applied (commit `5ad698b21`)
- CI run (iid=8753) with diffusion enabled + halo fix + all GPU flags → **non-zero diffs** (vn internal: 1.91e-10)

## CI configuration

- `LEVELS=validation` triggers `ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE=1` and `ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL=1`
- Integration level does NOT set these flags → non-zero default tolerances → inconclusive results

## Bisection results summary

| Round | Commit    | Change                                                         | vn internal max diff |
| ----- | --------- | -------------------------------------------------------------- | -------------------- |
| 1     | `31b950b` | Disable apply_diffusion_to_vn + temp diffusion                 | **0.0**              |
| 2     | `cb5aae5` | Re-enable apply_diffusion_to_vn only                           | **1.91e-10**         |
| 3     | `a2a59f8` | keep_skip_values=False                                         | NaN                  |
| 4     | `30d6d94` | Fix pentagon skip values                                       | 1.91e-10 (no effect) |
| 5     | `3a0b607` | Zero z_nabla4_e2 after computation                             | **0.0**              |
| 6     | `0ff505d` | Zero E2C2V inside \_calculate_nabla4, keep z_nabla2 arithmetic | **1.92e-11**         |

## What we know

Two independent sources of non-determinism in `_apply_diffusion_to_vn`:

1. **E2C2V Pattern B** (individual neighbor lookups: `E2C2V[0]`, `E2C2V[1]`, etc.) — contributes ~1.7e-10 (90%)
2. **Arithmetic in fused stencil** — contributes ~1.9e-11 (10%), even when E2C2V terms are zeroed and inputs are deterministic

Both run inside a fused `@gtx.field_operator` (`_apply_diffusion_to_vn`) that combines `_calculate_nabla4` + `_apply_nabla2_and_nabla4_global_to_vn` into a single GPU kernel.

Source (1) is explainable: GT4Py compiles the two E2C2V access patterns (broadcast vs indexed lookup) into different expression trees, leading to different operation orders on GPU even with FMA disabled.

Source (2) is surprising: the arithmetic in Round 6 is just `z_nabla4 = -8.0 * z_nabla2_e * (inv_dvv^2 + inv_dpe^2)` — all fields are wpfloat (float64), FMA is disabled, yet the fused stencil produces non-bitwise-identical output across MPI ranks.

## Working hypothesis

**Stencil fusion is the root cause.** GT4Py fuses `_calculate_nabla4` + `_apply_nabla2_and_nabla4_global_to_vn` into a single GPU kernel. The compiler may reorder operations across stencil boundaries (e.g., interleaving the nabla4 arithmetic with the vn update), and this reordering varies depending on domain size, register pressure, or other compilation factors that differ between single-rank and multi-rank runs.

If this hypothesis is correct, **separating the stencils into non-fused programs** should eliminate both sources of non-determinism.

## Investigation plan

### Phase 1: Test whether stencil fusion causes the arithmetic non-determinism

**Experiment 7: Separate `_calculate_nabla4` from the fused `_apply_diffusion_to_vn` stencil.**

Specifically: compile `calculate_nabla4` as its own `@gtx.program`, call it to write `z_nabla4_e2` into a pre-allocated buffer, then pass that buffer into a modified `_apply_diffusion_to_vn` field_operator that accepts `z_nabla4_e2` as input (instead of calling `_calculate_nabla4` internally).

Configuration:

- Keep Experiment 6's E2C2V zeroing (test ONLY the arithmetic, not E2C2V)
- Keep apply_diffusion_to_vn enabled, temperature diffusion disabled
- `LEVELS=validation`, `gtfn_gpu`

Expected outcomes:

- **Zero diffs** → stencil fusion IS the cause of both non-determinism sources. Proceed to Phase 2a.
- **Non-zero diffs (~1.9e-11)** → the arithmetic is non-deterministic even without fusion. Proceed to Phase 2b.

Code changes needed:

- Allocate `self.z_nabla4_e2` buffer (EdgeDim x KDim, dtype=vpfloat) in `Diffusion._allocate_local_fields()`
- Compile `calculate_nabla4` as a standalone program in `Diffusion.__init__`
- Modify `_apply_diffusion_to_vn` to take `z_nabla4_e2: EdgeKField[vpfloat]` as input, removing internal `_calculate_nabla4` call
- Call `self.calculate_nabla4(...)` in `run()`, then `self.apply_diffusion_to_vn(...)` with the pre-computed buffer
- Revert Experiment 5 and 6 bisection code (restore original \_calculate_nabla4, but called standalone)

### Phase 2a: If fusion IS the cause (Experiment 7 = zero diffs)

**Experiment 8: Restore full E2C2V computation with separated stencils.**

Configuration:

- Restore original \_calculate_nabla4 (E2C2V Pattern B) in the standalone program
- Keep stencils separated (from Experiment 7)
- Keep apply_diffusion_to_vn enabled, temperature disabled
- Expected: zero diffs (both sources eliminated by separation)

**Fix:** Keep stencils permanently separated. Commit clean version.

- Revert all bisection commits
- Convert E2C2V Pattern B → Pattern A (broadcast pattern, matches nabla2) for robustness
- Keep `calculate_nabla4` as standalone program
- Add `z_nabla4_e2` buffer to Diffusion

**Experiment 9: Enable temperature diffusion with fix.**

Configuration:

- Full diffusion (apply_diffusion_to_vn + temperature, with separated stencils)
- Expected: zero diffs

**Experiment 10: Test with dace_gpu backend.**

### Phase 2b: If fusion is NOT the cause (Experiment 7 still has diffs)

Then the arithmetic non-determinism is inherent — it persists even with non-fused stencils and deterministic inputs.

**Phase 2b-1: Test wpfloat-only path (no vpfloat).**

- Change `_calculate_nabla4` output to wpfloat
- Change `_apply_nabla2_and_nabla4_global_to_vn` to accept wpfloat z_nabla4_e2
- This tests whether the vpfloat→wpfloat cast at the stencil boundary introduces non-determinism

**Phase 2b-2: Fix E2C2V Pattern B → A anyway (regardless of arithmetic investigation).**

- This eliminates at least the 90% contribution
- If the remaining 10% cannot be eliminated, assess whether 1.9e-11 is acceptable

**Phase 2b-3: Test evaluation order in vn update.**

- Break `diff_multfac_vn * z_nabla4_e2 * area_edge` into separate assignment statements
- Use `as_contiguous` or explicit temporaries to force evaluation order

### Phase 3: Final integration and verification (gated on Phases 1-2)

- Revert all bisection code
- Commit clean fix (separated stencils + Pattern A)
- Enable full diffusion
- Test with `gtfn_gpu` and `dace_gpu`
- Run CI at `LEVELS=validation`

## Files that need changes

| File                                                                                                                 | Type of change                                                              |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/diffusion.py`                                     | Add `z_nabla4_e2` buffer, compile `calculate_nabla4` program, call sequence |
| `model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/calculate_nabla4.py`                     | Revert bisection code, optionally convert Pattern B → A                     |
| `model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/apply_diffusion_to_vn.py`                | Add `z_nabla4_e2` param, remove internal `_calculate_nabla4` call           |
| `model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/stencils/apply_nabla2_and_nabla4_global_to_vn.py` | Possibly change z_nabla4_e2 type (Phase 2b-1)                               |
