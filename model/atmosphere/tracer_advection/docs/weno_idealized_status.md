# WENO idealized experiments — status (2026-09-10 evening)

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

## Decisions

- Everything in **double** for now; single/mixed precision after icon4py PR #970 merges,
  in its spirit. Shape the code like #970 now (init-time numpy `float64`, boundary cast at
  state construction, config scalars `wpfloat` via `dataclass_scalars_to_wp`, `WP_EPS`,
  literals wrapped) — checklist in the session notes; apply in the driver-integration pass.
  Fortran `REAL(sp)` quantities (β in 103/132, hybrid residual) go behind one alias that
  currently resolves to working precision.
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
4. Dispersion relation (W4): his variants are in
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
