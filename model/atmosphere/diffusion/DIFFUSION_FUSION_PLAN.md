# Diffusion in one program — implementation plan

*icon4py diffusion · fusion of the second half of the diffusion granule*

## 1. Summary of findings

### 1.1 Current call sequence (`diffusion.py:834–1008`)

The `Diffusion.run()` method on `main` executes, after the two RBF interpolations and
their exchanges:

| # | Step | Program / call | Zone range (horizontal) | Vertical range |
|---|------|----------------|--------------------------|----------------|
| 1 | program | `calculate_nabla2_and_smag_coefficients_for_vn` | edges: LATERAL_BOUNDARY_LEVEL_5 .. HALO_LEVEL_2 | 0 .. nlev |
| 2 | program | `calculate_diagnostic_quantities_for_turbulence` (conditional: shear_type >= 1 or loutshs or a_hshr > 0) | cells: NUDGING .. LOCAL | 1 .. nlev (KHalf) |
| 3 | exchange | `z_nabla2_e` (EdgeDim) | | |
| 4 | program | `mo_intp_rbf_rbf_vec_interpol_vertex` (2nd call) | vertices: LATERAL_BOUNDARY_LEVEL_2 .. LOCAL | 0 .. nlev |
| 5 | exchange | `u_vert`, `v_vert` (VertexDim, full) | | |
| 6 | copy | `w` → `w_tmp` (`copy_field_on_cell_khalf`) | all cells | 0 .. nlev+1 (full KHalf allocation; the stencil test uses this wider range) |
| 7 | program | `apply_diffusion_to_vn` | edges: LATERAL_BOUNDARY_LEVEL_5 .. LOCAL | 0 .. nlev |
| 8 | exchange | `vn` (async start, `full_exchange=False`) → handle | | |
| 9 | program | `apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence` | cells: (NUDGING or LATERAL_BOUNDARY_LEVEL_4) .. HALO | 0 .. nlev (KHalf; production wiring `vertical_end = num_levels`, matching Fortran `jk = 1, nlev`) |
| 10 | wait | `halo_exchange_wait(handle)` | | |
| 11 | program | `calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools` (conditional: apply_to_temperature) | edges: NUDGING .. HALO | nlev-2 .. nlev |
| 12 | copy | `theta_v` → `theta_v_tmp` (`copy_field_on_cell_k`) | all cells | 0 .. nlev |
| 13 | program | `apply_diffusion_to_theta_and_exner` (conditional: apply_to_temperature) | cells: NUDGING .. LOCAL | 0 .. nlev |
| 14 | exchange | `theta_v`, `exner` (CellDim, conditional) | | |
| 15 | exchange | `w` (CellDim, conditional) | | |

### 1.2 Data flow between the five programs

- `calculate_nabla2_and_smag_coefficients_for_vn` produces `kh_smag_e` (limited,
  vpfloat), `kh_smag_ec` (raw, vpfloat — the pre-limiter value, kept only for the
  turbulence diagnostics) and `z_nabla2_e` (wpfloat).
- `z_nabla2_e` needs a halo exchange because the second RBF interpolation reads it
  on vertices (through V2E).
- `kh_smag_e` is read (a) in `apply_diffusion_to_vn` (through the `_apply_nabla2_…`
  inner operators, directly on edges — no halo needed for its own update domain,
  but it is produced only up to HALO_LEVEL_2), and (b) in
  `apply_diffusion_to_theta_and_exner` through `_calculate_nabla2_for_z`, which
  gathers `theta_v` through `E2C[0]`/`E2C[1]` — a cell read from the edge location,
  which is valid without exchange because the diffusion of theta_v writes only
  NUDGING..LOCAL and the gather reaches at most one halo ring of cells.
- `calculate_diagnostic_quantities_for_turbulence` reads pre-diffusion `vn` through
  C2E (i.e. halo edges of vn are read — valid since the dycore exchanged vn before
  diffusion runs) and `kh_smag_ec`, writes `div_ic`/`hdef_ic` on cells.
- The cold-pool program rewrites `kh_smag_e` in place on the last two levels; the
  following theta/exner program then consumes the enhanced value through
  `_calculate_nabla2_for_z` at those levels.
- `w_tmp`/`theta_v_tmp` exist because those programs read neighbour cells of the
  very field they write (`C2E2CO`/`C2E2C`/`E2C` gathers) — the copies freeze the
  pre-diffusion state. `vn` has the same hazard *today only through the
  diagnostics*, which read pre-diffusion vn; the vn update itself
  (`_apply_nabla2_and_nabla4_…`) reads neighbours only of `u_vert`/`v_vert`/
  `z_nabla2_e`, not of `vn`, so no vn copy is needed for the update itself.

### 1.3 The gt4py mechanisms this relies on

- **Per-output domains**: a `gtx.program` accepts `domain=(d1, d2, …)` — a tuple of
  domain dicts aligned with the `out=` tuple. Precedent:
  `model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/velocity_advection_predictor.py:395-444`.
- **The KDim equality bug**: `concat_where(dims.KDim == nlev-1, …)` is broken by
  gt4py's domain inference (https://github.com/GridTools/gt4py/issues/2205); the
  workaround, already cited at
  `model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/velocity_advection_terms.py:47-49`,
  is an inequality (`KDim < nlev-1` or `KDim >= nlev-2`).
- **Setup**: `setup_program` (`model/common/src/icon4py/model/common/model_options.py:146`)
  binds constant args and horizontal/vertical size args; `variants` selects
  compile-time specialisations.

### 1.4 Related prior work (all unmerged, all stale vs current main)

- Branch `d1-demote-kh-smag-ec` = PR #1478: fuses `calculate_nabla2_and_smag_…`
  with `calculate_diagnostic_quantities_for_turbulence` into
  `calculate_nabla2_smag_and_turbulence_diagnostics` using per-output domains,
  demotes `kh_smag_ec` from a state field to a program-internal temporary, turns
  the runtime diagnostics guard into a compile-time variant, and adds a stencil
  test. Merge-base with main: `11cddd359` (#1465).
- Branch `a6-a7-out-domain-w-diffusion` = PR #1479: replaces the two
  `concat_where` pass-throughs in the w diffusion with per-output domains (w:
  interior..halo; dwdx/dwdy: horizontal_start..horizontal_end on KHalf 1..nlev).
- PR #1481 (closed): naming sweep — `apply_X_to_Y` → `X_for_Y`, drop
  `calculate_` prefixes (e.g. `diffusion_for_vn`,
  `nabla2_smag_and_turbulence_diagnostics`).

The two closed branches are valuable as *design references* but should not be
rebased wholesale; this plan starts from current `main` and cherry-picks the ideas.

### 1.5 ICON Fortran reference (`mo_nh_diffusion.f90`)

- Diagnostics from pre-diffusion vn: lines 832–853 (`kh_c`, `div`, then `div_ic`,
  `hdef_ic`).
- Limiter order: `kh_smag_ec = kh_smag_e` (raw, :483), then
  `kh_smag_e = MAX(0, kh_smag_e − smag_offset)` (:485) and
  `MIN(kh_smag_e, smag_limit)` (:487).
- vn updates: :965–1112; vn sync at :1255 (icon-model) / :1271 numbering in the
  proposal (icon-exclaim checkout).
- Cold pool: :1315 (enh_diffu_3d on cells, `jk = nlev-1, nlev`, cell range
  `grf_bdywidth_c .. min_rlcell_int-1` = LATERAL_BOUNDARY_LEVEL_4 .. one row past
  LOCAL) and :1389/1427 (edge-side `MAX(kh_smag_e, enh_diffu_3d(E2C…))`, edge range
  `grf_bdywidth_e+1 .. min_rledge_int` = NUDGING_LEVEL_2 .. LOCAL — narrower than
  icon4py's current NUDGING .. HALO on both ends, which is the divergence to fix).
- Only other ICON consumer of `kh_smag_e` after the granule: moisture diffusion
  (`lhdiff_q`, declared in `mo_nonhydro_state.f90:2525-2537`); icon4py has no such
  consumer today.

## 2. The fusion plan

Goal: after the second RBF interpolation and its exchange, run **one** program
(`diffusion_for_prognostic_fields` — name per #1481 convention) that applies the
Smagorinsky limiter once at its consumer, computes the turbulence diagnostics from
pre-diffusion vn, updates vn, w (+ horizontal gradients), and theta_v/exner with
the cold-pool enhancement inlined, then exchange everything at the end.

### Proposed call sequence

1. **[program]** `mo_intp_rbf_rbf_vec_interpol_vertex` — unchanged (1st call)
2. **[exchange]** u_vert, v_vert — unchanged
3. **[program]** `calculate_nabla2_and_smag_coefficients_for_vn` — **changed**:
   publishes one *raw* `kh_smag` in vpfloat (no limiter, no `kh_smag_ec` output)
4. **[exchange]** z_nabla2_e — unchanged
5. **[program]** `mo_intp_rbf_rbf_vec_interpol_vertex` — unchanged (2nd call)
6. **[exchange]** u_vert, v_vert — unchanged
7. **[copy]** vn → vn_tmp (new `copy_field_on_edge_k`), w → w_tmp,
   theta_v → theta_v_tmp
8. **[program, fused]** `diffusion_for_prognostic_fields`:
   - limit kh_smag once (max(0, kh − smag_offset), then min(…, smag_limit));
   - turbulence diagnostics from `vn_tmp` (compile-time variant, as in #1478);
   - diffusion of vn (reads vn_tmp for safety/consistency with the copy pattern,
     writes vn);
   - diffusion of w + horizontal gradients (per-output domains per #1479);
   - cold-pool enhancement inline: `concat_where(dims.KDim >= nlev-2, maximum(kh_limited, max_over(enh_diffu_3d(E2C), axis=E2CDim)), kh_limited)`
     folded into the kh_smag expression consumed by `_calculate_nabla2_for_z`
     (the edge-side MAX's effective range becomes the closure of the theta/exner
     cell domain through E2C — see open question 3);
   - diffusion of theta_v and exner.
9. **[exchange]** vn, theta_v, exner, w — all at the end (the async vn exchange
   loses its overlap with the w diffusion; nothing reads vn after its last write
   in icon4py or in ICON).

`kh_smag_ec` disappears as a state field (and as an output of the producer).

### Step 01 — publish kh_smag raw

`_calculate_nabla2_and_smag_coefficients_for_vn` returns
`(kh_smag_raw_vp, z_nabla2_e)` only — drop the limiter lines and the second copy.
The limiter (max/min with `smag_limit`/`smag_offset`) moves into the fused
program, computed once at its only consumer. The initial-run special values of
`smag_limit`/`smag_offset` are passed as runtime args to the fused program, as
today.

### Step 02 — diagnostics move down

`div_ic`/`hdef_ic` become outputs of the fused program with per-output domain
`{CellDim: (nudging, local), KHalfDim: (1, nlev)}`, guarded by the
compile-time variant `compute_diagnostic_quantities: [True, False]` (the guard
expression `shear_type >= VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND or loutshs or
a_hshr > 0` is evaluated once in `__init__`, as #1478 does). They read `vn_tmp`
(the copy) and the raw `kh_smag` (replacing `kh_smag_ec`).

### Step 03 — vn exchange goes last

Remove the async vn exchange + wait pair around the w diffusion. After the fused
program, exchange vn (EdgeDim), theta_v/exner (CellDim, still conditional on
`initial_run or iforcing not in (NWP, AES)`), and w (CellDim, same condition).
vn's halo stays valid through the fused program because the dycore exchanged it
and diffusion only reads it via vn_tmp.

### Step 04 — four programs become one

The fused program `diffusion_for_prognostic_fields` composes the existing inner
field_operators and uses a per-output domain tuple aligned with
`out=(vn, w, dwdx, dwdy, theta_v, exner, div_ic, hdef_ic)`:

| output | domain |
|---|---|
| vn | `{EdgeDim: (lateral_boundary_5, local), KDim: (0, nlev)}` |
| w | `{CellDim: (w_start, halo), KHalfDim: (0, nlev)}` (production range; the stencil test may use `(0, nlev+1)` for the full KHalf allocation) |
| dwdx, dwdy | `{CellDim: (w_start, halo), KHalfDim: (1, nlev)}` |
| theta_v, exner | `{CellDim: (nudging, local), KDim: (0, nlev)}` |
| div_ic, hdef_ic | `{CellDim: (nudging, local), KHalfDim: (1, nlev)}` |

Verified against `diffusion.py:764-797` (`_determine_horizontal_domains`) and the
`setup_program` wiring at `diffusion.py:536-682`: `lateral_boundary_5` =
`_edge_start_lateral_boundary_level_5`, `local` = `_edge_end_local` /
`_cell_end_local`, `w_start` = `_horizontal_start_index_w_diffusion` (NUDGING for
limited areas, LATERAL_BOUNDARY_LEVEL_4 otherwise), `halo` = `_cell_end_halo`,
`nudging` = `_cell_start_nudging`, `interior` = `_cell_start_interior`.

The upper-damping-layer `concat_where` on w stays (it is a conditional
modification, not a pass-through). The cold-pool enhancement becomes the
`concat_where(KDim >= nlev-2, …)` branch described above — **must be an
inequality** (gt4py#2205). gtfn and dace restrict the C2E2C gather to those
levels; embedded computes it everywhere (wasted but safe).

### Supporting changes

- **`copy_field_on_edge_k`** — new generic program in
  `model/common/src/icon4py/model/common/math/stencils/generic_math_operations.py`
  (+ `_copy_field_on_edge_k` in `math/operators.py`), mirroring the cell variants,
  plus its unit/stencil coverage following the existing pattern.
- **`Diffusion` state**: drop `kh_smag_ec` from `_allocate_local_fields` and the
  integration-test assertions (`test_diffusion.py:209`); keep `kh_smag_e` renamed
  conceptually to "raw" (name can stay to limit churn, or become `kh_smag`).
- **Tests**:
  - Replace the four stencil tests (`test_apply_diffusion_to_vn.py`,
    `test_apply_diffusion_to_w_…py`, `test_apply_diffusion_to_theta_and_exner.py`,
    `test_calculate_diagnostic_quantities_for_turbulence.py`) with one stencil
    test for the fused program, parametrized on the compile-time variants
    (`compute_diagnostic_quantities`, `apply_to_temperature`-equivalent,
    `limited_area`, `type_shear`), following the #1478 test pattern. Inner
    operators keep their existing tests where they remain importable.
  - `test_diffusion.py` init test: drop the `kh_smag_ec` zero-assertion.
  - Datatests (`test_diffusion.py` main datatest, MPI `test_parallel_diffusion.py`)
    verify all outputs via `verify_diffusion_fields` — no change needed there.
- **Remove functions that only appear in test code**: after the rewiring, sweep
  the diffusion stencil modules for public `gtx.program` wrappers (and their
  private `_` field_operators, if nothing else imports them) whose only remaining
  importers are tests — delete both the function and its test. Verified today:
  no package outside `atmosphere/diffusion` imports from
  `icon4py.model.atmosphere.diffusion.stencils` (tmx has its own copies; tach
  forbids the cross-import), so the sweep is local to the component. Known
  candidate: `calculate_nabla2_for_theta` (currently zero production importers —
  its inner operators `_calculate_nabla2_for_z`/`_calculate_nabla2_of_theta` are
  imported directly by `apply_diffusion_to_theta_and_exner`). After the fusion
  deletes the four old program wrappers, re-run the same check for their inner
  operators before deleting anything that still has an import.
- **Naming**: follow #1481's convention for the new program name
  (`diffusion_for_prognostic_fields`); do not rename the old programs beyond
  deleting the fused-away ones.

### Implementation order (small, verifiable commits)

**Every step ends with the dace_cpu gate:** run the diffusion datatest on the
`dace_cpu` backend and only proceed when it passes (it exercises the fused
program through the full granule, which is exactly what the rewrite touches):

```bash
uv run --group test --frozen pytest --datatest-only -n0 \
    --backend dace_cpu \
    model/atmosphere/diffusion/tests/diffusion/integration_tests/test_diffusion.py \
    -k test_run_diffusion_single_step
```

(with `GT4PY_BUILD_JOBS=4`; the embedded backend needs no test data and runs as
`--datatest-skip` unit/stencil tests in the same step.)

1. Add `copy_field_on_edge_k` to `model/common` + tests.
   Gate: dace_cpu datatest.
2. Change the producer to publish raw kh_smag (drop `kh_smag_ec`); move limiter
   into a small helper; update its stencil test; drop the `kh_smag_ec`
   zero-assertion in the init test.
   Gate: dace_cpu datatest.
3. Create `diffusion_for_prognostic_fields` composing existing inner operators,
   per-output domains, inline cold pool (its effective edge range becomes the
   closure of the theta/exner cell domain through E2C — see open question 3) and
   the diagnostics variant; add its stencil test; rewire `run()` (copies, single
   program, end exchanges).
   Gate: dace_cpu datatest.
4. Delete the four old programs and their stencil tests; drop `kh_smag_ec` state;
   sweep for and remove functions whose only remaining importers are tests (see
   "Supporting changes" — start from `calculate_nabla2_for_theta`, then re-check
   the inner operators of the deleted programs).
   Gate: dace_cpu datatest.
5. Run the full verification ladder (below).

### Verification

- `uv run --group test --frozen pytest --datatest-skip model/atmosphere/diffusion/ -n auto` (with `GT4PY_BUILD_JOBS=4`)
- Stencil tests: `uv run --group test --frozen pytest model/atmosphere/diffusion/tests/diffusion/stencil_tests/ -n auto --datatest-skip`
- Datatests (serial): `uv run --group test --frozen pytest --datatest-only model/atmosphere/diffusion/ -n0`
- dace_cpu datatest (the per-step gate above, run once more on the final state):
  `uv run --group test --frozen pytest --datatest-only -n0 --backend dace_cpu model/atmosphere/diffusion/tests/diffusion/integration_tests/test_diffusion.py`
- MPI: `mpirun -np 4 .cscs-ci/scripts/ci-mpi-wrapper.sh uv run --group test --frozen pytest -v -s --with-mpi -n0 -k mpi_tests model/atmosphere/diffusion/`
- gtfn compile check for the fused program (the K-split edge field read through
  C2E has no repo precedent):
  `uv run --group test --frozen pytest --backend gtfn_cpu model/atmosphere/diffusion/tests/diffusion/stencil_tests/test_diffusion_for_prognostic_fields.py --datatest-skip -n0`
- Single precision: `ICON4PY_FLOAT_PRECISION=single uv run --group test --frozen pytest --datatest-skip model/atmosphere/diffusion/ -n auto`
- Pre-commit: `uv run --group dev --frozen --isolated pre-commit run --all-files`
- CSCS CI on the PR: `cscs-ci run default;MODEL_SUBPACKAGES=diffusion` (add `;BACKENDS=gtfn_cpu` for the compile risk).

## 3. Open questions / risks

1. **gtfn/dace compile risk (accepted)** — the K-restricted `max_over(enh(E2C))`
   feeding `concat_where(KDim >= nlev-2, …)` has no precedent in the repo. If
   gtfn fails, fall back to computing `enh_diffu_3d` everywhere (embedded
   semantics) and restricting only the select, or hoist the cold-pool max out of
   the theta branch into its own (cheap, K-limited) inner operator.
2. **Async vn exchange overlap is lost** — today the vn exchange overlaps the w
   diffusion; after fusion there is nothing left to overlap with. Only a
   measurement (benchmark / CSCS CI timing) can quantify the cost; the proposal
   accepts this.
3. **Cold-pool horizontal range divergence** — ICON applies the edge-side MAX on
   `grf_bdywidth_e+1 .. min_rledge_int` (NUDGING_LEVEL_2 .. LOCAL); icon4py currently
   uses NUDGING .. HALO (wider on both ends). The proposal says "fix the divergence in
   the same change". Inlining changes the mechanics: the MAX now happens inside the
   theta/exner output domain (cells NUDGING..LOCAL) for the E2C gather, so the
   effective edge range becomes the closure of that cell range through E2C —
   need to confirm this matches ICON's intent (the enhanced coefficient only
   feeds theta_v/exner diffusion in icon4py).
4. **Buffer ledger is a wash at best** — removing the vpfloat `kh_smag_ec` field
   but adding a wpfloat `vn_tmp` (edges ≈ 1.5× cells) is a trade, not a saving.
5. **Inlining forecloses ICON's moisture-diffusion consumer** — nothing in
   icon4py reads the enhanced coefficient, but if `lhdiff_q` is ever ported the
   coefficient must be published again. Leave a comment at the inline site.
6. **vn_tmp copy cost** — one extra edge-field copy per step; the functional
   granule that would retire all three copies is future work.
7. **`nlev` as a program arg** — `concat_where(KDim >= nlev-2, …)` needs `nlev`
   as a size arg (constant per run); check it is accepted as a compile-time
   vertical size by `setup_program` (it is passed today as `vertical_end`).

## 4. Key file paths

| File | Lines | Role |
|---|---|---|
| `model/atmosphere/diffusion/src/icon4py/model/atmosphere/diffusion/diffusion.py` | 799–1008 | `Diffusion.run()` call sequence to be rewritten |
| `…/diffusion.py` | 536–682 | `setup_program` wiring for the five programs |
| `…/diffusion.py` | 715–762 | `_allocate_local_fields` (kh_smag_e/ec, w_tmp, theta_v_tmp) |
| `…/stencils/calculate_nabla2_and_smag_coefficients_for_vn.py` | 16–114 | producer to be changed (raw kh_smag) |
| `…/stencils/apply_diffusion_to_vn.py` | 25–133 | vn update (inner operators stay) |
| `…/stencils/apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence.py` | 25–115 | w update (per-output domains per #1479) |
| `…/stencils/apply_diffusion_to_theta_and_exner.py` | 28–109 | theta/exner update (cold pool inlines here) |
| `…/stencils/calculate_diagnostic_quantities_for_turbulence.py` | 20–63 | diagnostics (move into fused program) |
| `…/stencils/calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools.py` | 20–61 | cold pool (to be inlined) |
| `…/stencils/enhance_diffusion_coefficient_for_grid_point_cold_pools.py` | 21 | edge-side MAX over E2C |
| `…/stencils/temporary_field_for_grid_point_cold_pools_enhancement.py` | 17–60 | enh_diffu_3d cell computation |
| `model/common/src/icon4py/model/common/math/stencils/generic_math_operations.py` | 87–129 | copy_field_on_cell_k/_khalf (pattern for edge variant) |
| `model/common/src/icon4py/model/common/math/operators.py` | 71–73 | `_copy_field_on_cell_k` (pattern) |
| `model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/velocity_advection_predictor.py` | 395–444 | per-output-domain precedent |
| `model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/velocity_advection_terms.py` | 47–49 | gt4py#2205 KDim-equality bug citation |
| `model/common/src/icon4py/model/common/model_options.py` | 146–201 | `setup_program` |
| `model/atmosphere/diffusion/tests/diffusion/integration_tests/test_diffusion.py` | 209–210 | kh_smag_ec init assertions to drop |
| `model/atmosphere/diffusion/tests/diffusion/utils.py` | 16–60 | `verify_diffusion_fields` (no change) |
| `icon-exclaim/src/atm_dyn_iconam/mo_nh_diffusion.f90` | 832–853, 965–1112, 1271, 1315, 1389 | Fortran reference |
