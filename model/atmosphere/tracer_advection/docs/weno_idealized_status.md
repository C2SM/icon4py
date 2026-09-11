# WENO idealized experiments — status (2026-09-11)

Companion to `weno_idealized_scope.md`. Branch `weno_idealized` (icon4py), capture branch
`transport_ajocksch_capture` (worktree `icon-ajocksch/`). Data under
`<project>/weno_data/` (grids, `reference/<case>/`, `reference/jocksch_grid/<case>/`,
build caches). Recipes: `icon-ajocksch/CAPTURE_NOTES.md`.

## Done

- **Fortran reference (Phase 0).** `transport_ajocksch` built with serialization and
  `-Kieee -Mnofma -gpu=nofma`; cylinder runs for 2/3/102/103/132/202/203 × hlim 0/3/4 and
  the three weight sets (opt, hand-tuned, ones) on (a) the generated torus
  `torus_20x22_res5000m[_centred].nc` and (b) **Andreas' own grid**
  `jocksch_torus_grid_r4_c200_elen100.nc` (from
  `/capstor/scratch/cscs/ajocksch/dispersion_relation/icon/grids/`). On (b) every entry of
  paper Table 2 reproduces; our build matches his own run to 2 ulp. Savepoints:
  `advection-init/exit` with `step=1..100`, `lsq-coefficients`.
- **Findings for Andreas.** (1) His cell-local PD limiter has no cell–edge orientation sign
  and only works on grids where every edge's normal flux is non-negative for the wind
  (true on his grid: `n_x ≥ 0` everywhere). (2) Paper's "d_j = 1" column is the all-ones
  weight set; the code's literal non-opt path `(1, 1.5, 1, 0.5, 1)` gives 3.278. (3) Hybrid
  selection uses the distance-weighted design matrix against unweighted values and unit
  run-time weights, so hybrid ≈ unit-weight WENO except in near-constant cells.
  (4) `max(q)` in eq. 6 is the cell value.
- **Python cylinder experiment** (`model/driver/tests/driver/integration_tests/test_jocksch_cylinder.py`,
  IC `initial_condition/analytical/moving_cylinder.py`): reproduces the Fortran printed
  error to 4–5 digits and Table 2 to printed digits for 2/3/102/103 and standard limiters
  (`gtfn_cpu`).
- **Scheme gaps ported** (commits `92a4cb3d8`, `6dbaa6496`, `84d71fc7f`, `f50ae5db1`,
  reviewed, no number-changing defects): weight sets `OPTIMIZED/HAND_TUNED/UNITY`, hybrid
  132, cell-local PD limiter (value 104, orientation-correct). Not yet wired into the driver.
- **Tests vs capture (W5, W5b, W5c).** Serialbox `step` key and lsq-coefficients reader,
  capture registered as a local experiment, L1/L2/trajectory test module (`c07baf6a4`),
  FMA conftest, Table 2 / Fortran gates, GPU sbatch script (`e49fdbd0b`, verified by W5b;
  review fixes and the hybrid 132 case in W5c). Measured on gtfn_cpu, dace_cpu, gtfn_gpu
  and dace_gpu (identical to the printed digits, GPU jobs 858221 / 858312): L1
  bit-identical except the SVD pseudoinverses (7e-13 / 2.5e-12; the interpolation
  factory's linear one 3.5e-16 on CPU, 7.4e-15 with cupy's SVD on GPU, harmless
  downstream); L2 per step (tracer 1; tracers 2-4 asserted bit-equal) 1e-15 for 2 and
  102, 2e-14 for 3, 2.5e-9 .. 3.5e-9 for 103 and 2.4e-9 for 132 (the Fortran's
  `REAL(sp)` smoothness indicator); trajectories grow over the first tens of steps and
  saturate (up to 220x the step-1 level where step 1 is one ulp, within 2x of the
  step-100 value everywhere). FMA off changes the last digit only (and makes the 102
  step-1 tracer bit-identical, as the GPU builds do); kept as tooling, not default. All
  eight cylinder gates pass (1e-11 .. 3e-10 to the Fortran pair sum, 3.5e-9 for 103).
  Tables, recipes and the GPU-job prerequisites (venv `cuda13` extra for the uenv's CUDA
  13, `uv` copy in `weno_data/bin/`, submit from the workspace root):
  `docs/running_the_jocksch_reference_tests.md`.
- **`REAL(sp)` shaping and the W3 review fixes (W3b–W3d: `a1fbc5fbb`, `553924346`,
  `e4f0f57c9`, `7fe68923e`, `0ef828663` and the W3d commits).** The Fortran's `REAL(sp)`
  quantities resolve to one alias, `common.type_alias.fortran_sp_float` (currently
  `wpfloat`; `fortran_sp_literal` = `gtx.float32` beside it for the unsuffixed real
  literals; bound at import, `set_precision()` does not rebind it — to be handled at the
  #970 merge). The 103/132 smoothness β path runs in it
  (`stencils/accumulate_weno_candidate_flux_weights.py`, f90 2643/2996-3008: `zlc`, `area`,
  the `2e0` literal and the dot product with `real(z_quad_vector_sum)` in the sp kind, the
  `1d-20` and the square in wp). Both cylinder modules (`test_jocksch_cylinder.py`,
  `test_jocksch_cylinder_jocksch_grid.py`) share `tests/driver/utils.run_cylinder_one_period`.
  Hybrid selection mask with the residual in sp vs wp (`test_miura_weno_hybrid_pipeline.py`):
  0 of 880 cells differ on the initial cylinder (step 0, both centres — decided by
  construction, every stencil there is constant or O(1)) and 0 at every step on the
  Fortran's evolved `advection-init` fields of both `ihadv132_hlim0` captures, steps
  1 / 2 / 10 / 50 / 100: on the original generated grid file, where the Fortran's
  cylinder around the origin is the 48-cell quarter-disc in the corner
  (`icon-ajocksch/CAPTURE_NOTES.md`, 'Grid'), WENO-selected cells 78 / 89 / 227 / 624 / 666
  of 880; on the `_centred` file (the paper's full 176-cell disc) 136 / 168 / 456 / 820 / 852.
  The assertion is measured per cell: single precision moves the residual by at most
  9.3e-6 of the cell's margin `|lsqe_wp − threshold|` (the 10-eps band of W3d was
  decorative — the residual `dot − z_b` cancels, so its perturbation scales with
  `eps_sp·|z_b|`, not with the residual; the smallest margin relative to the threshold is
  3.8e-3 on the quarter-disc at step 100 and 2.8e-2 on the full disc). FMA
  observation: `553924346` left every printed error digit of the reference and cylinder
  runs unchanged on gtfn_cpu but moved the last bit of the ~1e-14 relative mass change of
  the 103/132 cylinder rows (103: 8.209e-15 → 8.009e-15), because gcc contracts the new
  expression graph differently; with `-ffp-contract=off` the old and the new stencil are
  bit-identical end to end (7.208e-15 both; `weno_data/slurm/w3c_row103_{head,oldstencil}[_nofma].log`,
  `w3c_compare_accumulate.log`: 0 differing elements in every stencil output, contracted or
  not). "Arithmetic unchanged" in that commit message means the same operations in the
  same kinds, not the same contraction.

- **Dispersion relation θ = 0, schemes 2 and 3 (W4a;
  `tests/tracer_advection/integration_tests/test_jocksch_dispersion.py`, wave helper
  `common/initial_condition/analytical/plane_wave.py`).** Paper §4 / Figs. 5-6 and the
  θ = 0 half of Fig. 7 on Andreas' grid, mirroring his live block
  `icon-ajocksch/src/atm_dyn_iconam/mo_nh_stepping.f90:3344-3488` (wave `:3427-3437`,
  `dt = (a/2)·1.9999999999·CFL` `:3440`, `ω̃ = -ln(q_new/q_now)·i·(a/2)/dt` and `diff`
  `:3469-3478` at his cell 883/3 = 294 (0-based 293), raw principal branch — the older
  `MO_NH_STEPPING/mo_nh_stepping.f90_dispersion:2183` printed `Re ω̃ + 2π`, his tables do
  not). One α per level (51 levels, columns asserted independent bit-exactly), so the
  51 CFL × 51 α table is 102 `Advection.run` per scheme (0.01-0.05 s each once compiled,
  ~20 s granule setup; full table ~25 s per scheme on gtfn_cpu). Against
  `dispersion_linear.txt` / `dispersion_quadratic.txt`: scheme 2 max |Δ Re ω̃| 1.5e-14,
  |Δ Im ω̃| 4.9e-14 (gates 5e-14 / 1.5e-13); scheme 3 1.2e-11 / 6.3e-11 (gates 4e-11 /
  2e-10; the L1 SVD round-off 7e-13 amplified by the logarithm, worst at CFL 0.48);
  dace_cpu (CFL 0.2) gives the same digits (scheme 2 bit-identical to gtfn_cpu on every
  cell, scheme 3 within 2.5e-16 in the tracer). Parity: cell 294 is tip-up; cells of one
  parity ≥ 3 edge lengths from the periodic seam agree to 5e-13 / 2e-13 (scheme 2, Re
  modulo the aliasing period `2π(a/2)/dt` — the principal branch flips by round-off near
  `2·CFL·α = π` — / Im) and 1.4e-11 / 8e-11 (scheme 3), the two parities' means to
  9e-16 / 4e-16 and 5e-14 / 2.4e-13 — the sampling cell does not matter at θ = 0. The
  wave is periodic on the torus only for ireal a multiple of 10, so the seam at
  x = 47.5 ↔ −52.5 km contaminates the cells within stencil reach: 66 cells (3 columns)
  for scheme 2, 110 (5 columns) for scheme 3; his cell is 7 cells away.
  Growth (−Im ω̃ > 0): scheme 2 from CFL 0.70 (α ∈ (0, 1.27), all α from CFL 0.76, max
  rate 0.35 at CFL 1), scheme 3 from CFL 0.62 (α ∈ (0, 0.38) → (0, 1.80) at CFL 1, rates
  < 4.4e-3); nothing grows for the six CFLs of the paper. Outputs in
  `weno_data/dispersion/`: `ihadv{2,3}_theta0.txt` (his 4-column F20.16 format), per-cell
  `_cells.npz`, `_summary.txt` (per-CFL Δ vs Fortran, parity spread, seam cell count,
  growth boundary), `fig5/6/7_*.png` (icon4py solid, Fortran dashed, exact black). The
  `_setup` there (granule from `driver.initialize_driver` on the grid file, unit air
  mass, `prescribe_uniform_wind`) is the harness W4c (θ = 30°) extends.

## Decisions

- Everything in **double** for now; single/mixed precision after icon4py PR #970 merges,
  in its spirit. Shape the code like #970 now (init-time numpy `float64`, boundary cast at
  state construction, config scalars `wpfloat` via `dataclass_scalars_to_wp`, `WP_EPS`,
  literals wrapped) — checklist in the session notes; apply in the driver-integration pass.
  Fortran `REAL(sp)` quantities (β in 103/132, hybrid residual) sit behind one alias,
  `type_alias.fortran_sp_float`, that currently resolves to working precision.
- Milestone 1 = idealized experiments in Python; no granule/binding (that is milestone 3,
  after FFSL).

## Next

1. Driver integration: `driver_utils` passes `l_weights_s` per config, builds the hybrid
   state, wires limiter 104; review items (butterfly mask dtype/bool, E2C-orientation
   assertion in the limiter, hybrid-state coupling check, citation drift); #970 shaping.
2. Python cylinder on Andreas' grid for all Table 2 rows incl. hybrid, UNITY, the four
   limiter rows; compare with `reference/jocksch_grid/`.
3. W5 follow-ups: gates on all backends (`dace_gpu` > `dace_cpu` > `gtfn_gpu`), 103 order
   study on the torus patch.
4. Dispersion relation: W4b (capture of the θ = 0 block from our build, L2 per (CFL, α))
   and W4c (θ = 30°, the chequerboard re-imposition loop of `mo_nh_stepping.f90:3489-3700`,
   `dispersion_ffsl_30.txt`); his variants are in
   `/capstor/scratch/cscs/ajocksch/dispersion_relation/MO_NH_STEPPING/`.
5. Milestone 2: FFSL (4/5/22/32/42/52), PSM, vlimit 2/3.

## Sandbox notes (husk)

`uv` needs `UV_PYTHON=$PWD/.venv/bin/python UV_PYTHON_INSTALL_DIR=$TMPDIR/uvpy
UV_CACHE_DIR=<project>/weno_data/uv_cache`; gt4py needs
`CXX=/user-environment/env/default/bin/g++ CC=…/gcc`; one `GT4PY_BUILD_CACHE_DIR` per
backend; pre-commit needs `GIT_HTTP_PROXY_AUTHMETHOD=basic PRE_COMMIT_HOME=<writable>`;
git identity via `GIT_AUTHOR_*`/`GIT_COMMITTER_*`; `source env.sh`/`setup.sh` die on
seccomp (`set +o privileged`) — use `icon-ajocksch/build_serialize/build.sh`; SLURM only
`--partition=debug`, env set inside job scripts; one pytest at a time.

## Experiment details worth knowing (established 2026-09-10)

- Normal convention: icon4py's `EDGE_NORMAL_U/V` (primal normal, cell 1 → cell 2) has dot
  product +1 with Andreas' edge-vector construction `(-Δy, Δx)/|e|` on all 1320 edges of the
  generated grid, so `mass_flx_me = u·n_x + v·n_y` is his flux.
- dt is exactly 1000 s in Python; his `(a/2)·1.9999999999·CFL` is 999.99999995 s (5 µm of
  displacement over 100 steps).
- His pair sum excludes pairs whose raw centre distance is ≥ 5000 m (i.e. across the periodic
  wrap); the Python test prints both that and the all-pairs sum (= 3·Σe²); they differ at 1e-4.
- Five vertical levels (the metrics factory fails below three); columns are identical.
- On Andreas' grid `mass_flx_me ≥ 0` on every edge (8800 positive, 4400 zero) — the condition
  his cell-local PD limiter needs. On the generated grid 440 slanted edges have `vn < 0`.
- The paper's Table 2 numbers are `sqrt(Σe²)`; his printed `#` number is the pair sum (3·Σe²);
  the paper truncates rather than rounds.
- Fortran reference sets: `weno_data/reference/<case>/` (generated grid; `_centred` = cylinder
  at (L/2, H/2), which is what the Python default centre reproduces) and
  `weno_data/reference/jocksch_grid/<case>/` (his grid, definitive; includes `_dj1` =
  hand-tuned weights and `_ones` = paper's d_j = 1).
