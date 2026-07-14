# LOG

## 2026-07-13 Session start

- Reloading build skill, resuming coordinator role
- Step 1: Spawning build-explorer to compare diffusion vs ICON Fortran reference

## 2026-07-13 CI Round 1

- Experiment 1 committed (31b950bd0)
  - Disabled: apply_diffusion_to_vn, enhanced diffusion, apply_diffusion_to_theta_and_exner
  - Kept: RBF interp 1+2, calculate_nabla2_and_smag, z_nabla2_e exchange, w diffusion
  - CI triggered: BACKENDS=gtfn_gpu;LEVELS=integration;GT4PY_BUILD_CACHE_LIFETIME=session
  - Waiting for results...

## CI Round 1 update

- Initially pushed to origin (C2SM/icon4py) — wrong remote! PR head is msimberg/icon4py
- Re-pushed to msimberg remote at 16:55
- Re-posted CI comment at 16:55
- Polling for CI pipeline start (background task bash-ecad8dc2, 2min intervals)

## CI Round 1 rerun (validation level)

- Previous CI run (iid=8758) used LEVELS=integration → ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE and ICON4PY_DALLCLOSE_PRINT_INSTEAD_OF_FAIL were NOT set → test passed with non-zero default tolerances (atol=1e-10) → result inconclusive
- Root cause: ci/base.yml:175 only sets reproducibility flags when \$LEVEL == 'validation'
- Posted new CI comment with LEVELS=validation (comment URL: https://github.com/C2SM/icon4py/pull/1368#issuecomment-4961222739)
- Waiting for pipeline to start and complete

## CI Round 1 result (validation level)

- Pipeline iid=8761 child job completed: 1 passed in 2050s (34 min)
- ALL ZERO DIFFS: internal max diff 0.0, HALO max diff 0.0, HALO_LEVEL_1 max diff 0.0, HALO_LEVEL_2 max diff 0.0
  - Confirmed for all 5 fields (vn, w, exner, theta_v, rho) with zero tolerance (atol=0, rtol=0)
- Conclusion: apply_diffusion_to_vn and temperature diffusion are the sources of non-determinism
- Proceeding to Round 2: re-enable apply_diffusion_to_vn only

## CI Round 2

- Re-enabled apply_diffusion_to_vn only, temperature diffusion still disabled (commit cb5aae595)
- Pushed to msimberg remote, CI comment posted
- Pipeline iid=8763 pending
- Expected: vn diffs should appear (apply_diffusion_to_vn internally uses calculate_nabla4 with direct E2C2V access)

## CI Round 2 result

- Pipeline iid=8763/8764: 1 passed in 2044s (34 min)
- apply_diffusion_to_vn re-enabled → NON-ZERO DIFFS in vn:
  - vn internal max diff: 1.91e-10, HALO max diff: 7.68e-11
  - theta_v internal max diff: 1.79e-10, HALO max diff: 6.46e-11
  - w internal max diff: 1.77e-13 (downstream)
  - exner internal max diff: 3.62e-14 (downstream)
  - rho internal max diff: 9.92e-13 (downstream)
- Confirms: apply_diffusion_to_vn introduces vn non-determinism; other field diffs are downstream
- More analysis: calculate_nabla2_and_smag uses u_vert_wp(E2C2V) with full dimension broadcast → all 4 E2C2V neighbors are accessed per edge; if any are INVALID_INDEX, garbage propagates
- Proceeding to Round 3: keep_skip_values=False

## CI Round 3

- keep_skip_values=False (commit a2a59f87d) replaces INVALID_INDEX in all neighbor tables with max valid neighbor
- apply_diffusion_to_vn ENABLED, temperature diffusion DISABLED (same as Round 2 config)
- Expected: if E2C2V skip values are the cause → zero diffs; if not → still non-zero (issue elsewhere)
- CI triggered, waiting for pipeline

## CI Round 3 result

- Pipeline iid=8766/8767: 1 passed in 1986s (33 min)
- keep_skip_values=False: ALL NaN diffs!
- Root cause: \_should_replace_skip_values replaces skip values for ALL dimensions on distributed grids, including CONNECTIVITIES_ON_PENTAGONS (V2EDim, V2CDim, V2E2VDim)
  - Pentagons have 5 valid entries in a 6-entry table; replacing the 6th entry with a valid index breaks stencil → NaN
- Fix: \_should_replace_skip_values should only replace for CONNECTIVITIES_ON_BOUNDARIES (halo skip values), not CONNECTIVITIES_ON_PENTAGONS

## CI Round 4

- Fixed \_should_replace_skip_values (commit 30d6d9471) to exclude pentagon skip values
- keep_skip_values=False still active (from Round 3)
- apply_diffusion_to_vn ENABLED, temperature diffusion DISABLED
- Expected: zero diffs (E2C2V boundary skip values replaced, pentagon skip values preserved)
- CI triggered, waiting for pipeline

## CI Round 5

- Zero out z_nabla4_e2 in \_apply_diffusion_to_vn (commit 3a0b607fa)
- Reverted keep_skip_values/icon.py changes (no effect from Round 4)
- apply_diffusion_to_vn ENABLED (nabla4 disabled, smag only), temperature DISABLED
- If zero diffs → \_calculate_nabla4 is the source
- If non-zero diffs → smag term (kh_smag_e * z_nabla2_e) is the source
- CI triggered, waiting

## CI Round 5 result

- Pipeline iid=8772/8773: 1 passed in 1948s (32 min)
- Zero z_nabla4_e2 AFTER compute → ALL ZERO DIFFS
- Confirmed: \_calculate_nabla4 introduces the non-determinism

## CI Round 6

- Zero E2C2V-based nabv_tang_vp and nabv_norm_vp INSIDE \_calculate_nabla4 (commit 0ff505dc6)
- z_nabla4_e2 computed using only z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length (no E2C2V)
- apply_diffusion_to_vn ENABLED (full), temperature DISABLED
- If zero diffs → E2C2V access in \_calculate_nabla4 is the source
- If non-zero → issue is in z_nabla2_e-dependent part of \_calculate_nabla4 or in \_apply_nabla2_and_nabla4_global_to_vn
- CI iid=8775, waiting

## CI Round 5 result

- Pipeline iid=8772/8773: 1 passed in 1948s (32 min)
- Zero z_nabla4_e2 AFTER compute via `z_nabla4_e2 = z_nabla4_e2 * 0.0` → ALL ZERO DIFFS
- Confirmed: \_calculate_nabla4 introduces the non-determinism

## CI Round 6 result

- Pipeline iid=8775/8776: 1 passed in 1962s (33 min)
- Zero E2C2V terms INSIDE \_calculate_nabla4, keep z_nabla2-e-dependent arithmetic
- z_nabla4_e2 computed from only z_nabla2_e + geometry (E2C2V terms zeroed)
- Result: NON-ZERO diffs (vn internal: 1.92e-11, theta_v: 3.21e-11, w: 1.22e-13, rho: 1.91e-13, exner: 4.44e-15)
- Diffs ~10x smaller than Round 2 (1.91e-10) but still non-zero
- Key insight: TWO distinct sources — E2C2V lookups (~90%) + simple vpfloat arithmetic (~10%)
- Even the fused z_nabla2-e arithmetic produces non-deterministic vpfloat output

## Session 2026-07-14 16:00

- Co-ordinator session: loaded build skill, analyzed current state
- Current branch state: Round 6 results show BOTH E2C2V lookups and z_nabla2-e arithmetic contribute to non-determinism
- The z_nabla2-e arithmetic contribution is hard to explain — inputs are deterministic (z_nabla2_e halo-exchanged, geometry static)
- Possible explanations:
  a. vpfloat/wpfloat cast in fused stencil introduces rank-dependent rounding
  b. GT4Py stencil fusion reorders operations from \_calculate_nabla4 + \_apply_nabla2_and_nabla4_global_to_vn
  c. Multiplication of 5 edge fields in the vn update amplifies sub-ULP differences
  d. ghex halo exchange produces different buffer alignments on different ranks

## CI Round 7

- Commit: `30cf06a5f` ("BISECT exp 7: separate calculate_nabla4 from the fused apply_diffusion_to_vn stencil")
- Reverted Experiment 5 code (no longer need to zero z_nabla4_e2 after computation)
- Keeps Experiment 6's E2C2V zeroing inside \_calculate_nabla4 (test arithmetic only, not E2C2V)
- Changes:
  - `_apply_diffusion_to_vn` field_operator: removed u_vert, v_vert, geom params; now accepts z_nabla4_e2 as input
  - `calculate_nabla4` compiled as standalone program (not fused)
  - `self.z_nabla4_e2` buffer allocated in Diffusion
  - Call order: halo exchange van/vert → `calculate_nabla4` → `apply_diffusion_to_vn`
- CI triggered: `LEVELS=validation`, `gtfn_gpu` (comment: https://github.com/C2SM/icon4py/pull/1368#issuecomment-4971163907)
- If zero diffs → stencil fusion is confirmed as the root cause
- If non-zero diffs → non-determinism persists even without fusion → investigate wpfloat/vpfloat casting or evaluation order
- Waiting for pipeline...

## CI Round 7 result

- Pipeline iid=8783/8784: 1 passed
- Result: NON-ZERO diffs — IDENTICAL to Round 6 (vn internal: 1.92e-11, theta_v: 3.21e-11)
- **Key finding: stencil fusion is NOT the cause.** Separating `calculate_nabla4` from `_apply_diffusion_to_vn` (writing result to a buffer, reading in separate program) produces exactly the same diffs as the fused version.
- The non-determinism is inherent — it persists across memory boundaries, not just within a fused kernel.

## CI Round 8

- Commit: `535bfe34` ("drop z_nabla4_e2 contribution from vn update")
- Removed `diff_multfac_vn * z_nabla4_e2_wp * area_edge` from `_apply_nabla2_and_nabla4_global_to_vn`
- Keep `calculate_nabla4` running but discard its output in the vn update
- vn update simplifies to: `vn += area_edge * kh_smag_e * z_nabla2_e`
- Tests whether the non-determinism is in the z_nabla4_e2 * area_edge multiplication chain
- If zero diffs → confirms z_nabla4_e2 is the source; issue is in multiplication
- CI triggered: `LEVELS=validation`, `gtfn_gpu` (comment: https://github.com/C2SM/icon4py/pull/1368#issuecomment-4971631457)
- Pipeline iid=8786 pending; monitoring for child pipeline results
