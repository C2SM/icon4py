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
  θ = 0 half of Fig. 7 on Andreas' grid, mirroring his block (lines of the capture branch
  at `icon-ajocksch` commit `db7a1f149d`, `src/atm_dyn_iconam/mo_nh_stepping.f90:3397-3550`;
  pristine `dacecf46aa:3322-3466`: wave `:3488-3495`, `dt = (a/2)·1.9999999999·CFL`
  `:3498`, `ω̃ = -ln(q_new/q_now)·i·(a/2)/dt` and `diff` `:3534-3543` at his cell 883/3 =
  294 `:3528` (0-based 293), raw principal branch — the older
  `MO_NH_STEPPING/mo_nh_stepping.f90_dispersion:2183` printed `Re ω̃ + 2π`, his tables do
  not). One α per level (51 levels, columns asserted independent bit-exactly), so the
  51 CFL × 51 α table is 102 `Advection.run` per scheme. Granule setup: 136-213 s for the
  first scheme of a pytest process on the login node (the programs compiled or loaded
  from the cache: gtfn_cpu 180 / 213 s, dace_cpu 136 s), 12-17 s for the second; the full
  table then takes 5 s (scheme 2) / 6 s (scheme 3) of runs (W4a's summaries: total 185 s /
  23 s). Against
  `dispersion_linear.txt` / `dispersion_quadratic.txt`: scheme 2 max |Δ Re ω̃| 1.5e-14,
  |Δ Im ω̃| 4.9e-14 (gates 5e-14 / 1.5e-13); scheme 3 1.2e-11 / 6.3e-11 (gates 4e-11 /
  2e-10; the L1 SVD round-off 7e-13 amplified by the logarithm, worst at CFL 0.48);
  dace_cpu (CFL 0.2) gives the same digits (scheme 2 bit-identical to gtfn_cpu on every
  cell, scheme 3 within 2.5e-16 in the tracer); since W4c the `single` set asserts its
  table at cell 294 against the saved gtfn_cpu full table at 3e-15 (dace_cpu, job 859997:
  5.6e-17 / 5.6e-17 for scheme 2 — the F20.16 print — and 4.4e-16 / 1.1e-15 for scheme 3;
  untested on the GPU backends), and fig. 7's colormap is centred at zero. The θ = 0
  tables of the W4c rerun (job 860014) are byte-identical to W4a's. Parity: cell 294 is
  tip-up; cells of one
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

## W4c — dispersion relation θ = 30°, schemes 2 and 3 (paper Fig. 7, 2026-09-11)

`tests/tracer_advection/integration_tests/test_jocksch_dispersion.py::test_dispersion_relation_theta30`
(sets `paper`: CFL 0.01, 0.1, …, 0.5, gated against his tables; `stability`: 0.42 / 0.44;
`env`: job-script lists), helpers `plane_wave.chequerboard_phase_factor` /
`reimpose_chequerboard_wave`. Mirrors Andreas' chequerboard block, lines of
`icon-ajocksch` `db7a1f149d:src/atm_dyn_iconam/mo_nh_stepping.f90` (pristine
`dacecf46aa:3467-3678`; this block produced his `linear_oblique.txt_new` = scheme 2 and
`quadratic_oblique5.txt` = scheme 3 and our build's `dispersion_ihadv{2,3}_theta30_full`):

- wind at `angle = dble(30)/180d0*acos(-1d0)` `:3567-3572` as his edge-vector mass flux
  `-y_comp/length_ref·u_x + x_comp/length_ref·u_y` `:3573-3641` (vertex 2 − vertex 1 with the
  periodic-wrap replacements, the `0.1`/`1.1` default-real literals kept; `length_ref` =
  edge 695 = 5000.0 exactly), computed from icon4py's vertex coordinates and E2V and used
  for `mass_flx_me` and `vn_traj` on all levels `:3643`; it equals `u·n` of the geometry to
  8.9e-16 (asserted at 1e-12, which fixes the vertex order: a reversed order gives O(1));
  air mass 1 `:3646-3647`;
- α = ireal/100·π, ireal = 0, 2, …, 116, one per level (59) `:3569-3570`; the wave
  `exp(-iα(x u_x + y u_y)·2/a)` once per CFL `:3651-3672`;
- 1000 iterations `:3678` of {`Advection.run` on the real and on the imaginary tracer with
  `dt = (a/2)·1.9999999999·CFL` `:3679`; the capture's convergence record at cell 695
  `:3703-3713` (`n_last` = last iteration with |Δω| ≥ 1e-12, 1 at the first; `resid` = |Δω|
  after the last); except in the last iteration `:3718`, the wave re-imposed from the
  advected field `:3731-3751`: cell j (1-based) gets `tracer_ref(mod(j,2)+1)·exp(-i((x_j −
  x_ref)u_x + (y_j − y_ref)u_y)·2/a·α)`, real and imaginary part each divided by
  `|tracer_ref(1)|`, reference 2 = q_new(695), reference 1 = q_new(696)}; then ω at 695 and
  ω₂ at 696 `:3770-3779` from the last imposed wave and the last advected field. 0-based
  cells 694 / 695; on his grid the index parity is the triangle orientation for all 880
  cells (694 tip-down, 695 tip-up; asserted). The complex product of the re-imposition is
  written out as `(ac − bd, ad + bc)` (the `-Mnofma` build's product; numpy's complex
  multiply is FMA-contracted on aarch64 and differs in the last bit); the vectorised
  re-imposition is bit-identical to a literal loop transcription of `:3714-3753`.
- outputs in `weno_data/dispersion/`: `ihadv{2,3}_theta30.txt` (six CFLs) and
  `ihadv{2,3}_theta30_full.txt` in his layout (`6F25.16`, blank line between CFL blocks,
  `:3780-3783`), `…_conv.txt` (the `#conv` lines `(a,F25.16,F25.16,i8,es14.3)` `:3784-3787`,
  plus `#first` = first iteration with |Δω| < 1e-12, per-CFL wall), `…_summary.txt` (per-CFL
  comparison and statistics), per-CFL `theta30_parts/<set>_<backend>/ihadv<s>_cfl<c>.npz`
  (ω, conv record, early-stop value, final `q_now`/`q_new` on all cells),
  `fig7_ihadv{2,3}_theta30_full_run_gtfn_cpu.png` and `fig7_ihadv{2,3}_theta0_theta30_…png`
  (θ = 0 and θ = 30 panels side by side; −Im ω̃ over (α, CFL), diverging map centred at 0,
  zero contour, θ = 30 non-converged rows grey).

Runs (gtfn_cpu, compute node, debug jobs owning the lock, `weno_data/slurm/w4c_dispersion.sbatch`):
job 859994 `paper` + `stability` (both schemes, 232 s pytest), job 859995 the full 51 CFL ×
59 α tables for both schemes (799 s pytest; 3.7 s / 9.4 s per CFL for scheme 2 / 3, i.e.
2000 `Advection.run` at 1.5 / 4.4 ms each, setup 22 / 15 s), 859996 its continuation (found
all CFLs saved, rewrote the outputs), 860014 the final check (θ = 0 six tests, θ = 30
gated sets: 10 passed). The paper and stability CFLs of jobs 859994, 859995 and 860014 are
bit-identical (ω, conv record, early-stop values, final fields; the first run's parts kept
in `theta30_parts/*_job859994/`).

Comparison ("converged" = resid ≤ 1e-8; max |Δ| over the four ω columns):

| scheme | vs his, converged | vs his, all 3009 rows | vs our build, converged in both | vs our build, all rows |
|---|---|---|---|---|
| 2 | 1.9e-14 (CFL 0.01; 2.0e-15-3.5e-15 at 0.1-0.5) | 2.3e-12 | 1.7e-14 | 3.4 (the two α = 0 rows at CFL 0.46/0.48, below; else ≤ 2.2e-12) |
| 3 | 1.8e-14 (CFL 0.01; 2.2e-15-2.7e-15 at 0.1-0.5) | 3.4e-12 | 1.8e-14 | 2.2e-12 |

Gates against his tables, converged rows, max over the set: 6e-14 on the `paper` set (3× its
CFL 0.01 block, 17× / 22× the CFL 0.5 block for scheme 2 / 3) and on the `stability` set
(measured 4.4e-15 / 3.0e-15 at CFL 0.44); gtfn_cpu measurements only. Without his tables (his
scratch is purgeable) both sets skip with the reason, the `stability` set after its growth
asserts, and the summaries print n/a. Stability limit:
identical to our build and to his tables — no growth (Im ω < −1e-14 on converged rows,
either cell) through CFL 0.42, growth from 0.44: scheme 2
32 of 33 converged α, min Im ω −3.609e-2 at α 3.644 (0.50: 56/56, −0.2696); scheme 3 15 of
58, −1.381e-5 at α 3.267 (0.46: −3.452e-4 at 0.628; 0.50: 40/57, −3.547e-3 at 2.639); the
minima agree with our build's to all printed digits. Convergence after 1000 iterations
(rows with resid > 1e-12 / 1e-8 / 1e-4): scheme 2 222 / 154 / 89 (our build 224 / 156 / 91),
scheme 3 153 / 100 / 57 (identical); `n_last` identical in 2996 / 2992 of 3009 rows, ±1
elsewhere among rows converged in both; medians 589 / 74 / 46 / 58 (CFL 0.01 / 0.1 / 0.2 /
0.3) and 913 at 0.44 for scheme 2, 859 / 98 / 44 / 35 for scheme 3. The only different
branch: scheme 2 α = 0 at CFL 0.46 and 0.48 stays at ω = 0 in icon4py (resid 0, the
constant is kept exactly) like in his table, while our build flips to the chequerboard mode
q_new ≈ −q_now (Re ω = π/(1.9999999999·CFL)); all three flip at 0.50. Parity spread on
converged rows, max |ω(695) − ω(696)|: 5.8e-8 (scheme 2) / 4.0e-7 (scheme 3), at CFL 0.01,
as our build's; ≤ 7.2e-9 / 1.0e-7 from CFL 0.1.

Early stop: stopping a level at the first |Δω| < 1e-12 would change ω by up to 2.0e-11 /
3.8e-11 at CFL 0.01 and 5e-13-2.3e-12 at CFL 0.1-0.5 (bit-identical only for α = 0), and at
CFL 1.0 scheme 3 α = 0 by 0.22 (the constant converges at iteration 2, the growing mode
takes over later) — so every level runs his 1000 iterations; the non-converged symmetry
level α = 1.822 needs them anyway.

**Sensitivity to the pseudoinverse (hypothesis test for findings #8).** At θ = 0, icon4py
and our build differ in scheme 3 at his cell 294 (0-based 293) by |Δω|·2·CFL·|q_new/q_now|
(the round-off of the one-step ratio; |q_new/q_now| = exp(−1.9999999999·CFL·Im ω̃)) =
9.8e-15 / 1.3e-13 / 3.1e-13 / 5.0e-13 / 6.8e-13 / 8.1e-13 at CFL 0.01 / 0.1 / 0.2 / 0.3 /
0.4 / 0.5, attributed to the SVD pseudoinverses (L1: 7.2e-13 relative). The same two
implementations at θ = 30, converged rows, max over both cells: 3.6e-16 / 5.0e-16 / 9.4e-16 /
1.1e-15 / 1.6e-15 / 2.6e-15 — 27-470× smaller, and at the level of scheme 2 (3.3e-16-1.9e-15,
whose pseudoinverses differ by 3.5e-16); max over all converged rows with CFL ≤ 0.5: 4.3e-15
(component) / 5.5e-15 (modulus), 3.7e-15 on rows with resid < 1e-12 in both. Our build vs his
`quadratic_oblique5.txt` on the same rows: ≤ 4.7e-15.

The contrast is the sampling cell, not the angle: the pseudoinverse round-off concentrates in
strips of cells (W4c review, numpy, `weno_data/w4c_review/percell.py` and `pinv.py`).
icon4py vs our build at θ = 0, scheme 3, the same run from the same q_now, per cell over the
638 cells ≥ 3 edge lengths from the x seam, same normalisation: cell 293 lies in the worst
strip at every CFL (9.8e-15 … 8.1e-13; within 1.1 % of the maximum at CFL 0.01 and 0.1 %
from CFL 0.1); cells 694 / 695 (his θ = 30 cells 695 / 696) stay at the θ = 30 level,
2.2e-16 … 1.05e-15 over CFL 0.01 … 0.5; the median cell 2.2e-16 … 2.1e-15. Only whole strips
exceed 1e-13: the rows at y = 14.43 / 15.88 km (all 40 cells, containing 293) from CFL 0.1,
and y = −37.5 / −36.1 km from CFL 0.4 (29 of the 638 cells to CFL 0.3, 58 from 0.4). The
−37.5 / −36.1 km rows are a strip at every CFL, uniform along the row (6.0e-14 at CFL 0.2,
9.98e-14 at 0.3, 1.36e-13 at 0.4; within 0.9 % over their 29 interior cells from CFL 0.1),
that crosses 1e-13 at 0.4. Our build's own `lsq_high_lsq_pseudoinv` (capture) deviates from
the median over its triangle orientation, relative to the largest entry, by 7.4e-13 (tip-up) /
3.6e-13 (tip-down) in the strip of 293 and by ~2e-15 / 4e-15 in the rows of 695 / 694 (median
cell 1.8e-15 / 2.6e-15). icon4py's own θ = 0 field shows the 14.43 km strip too: against the
strip-free value (the median over the same-orientation interior cells outside the two strips;
normalised as above, max over α) its strip cells deviate by up to 2.8e-13, our build's by up
to 5.3e-13 at CFL 0.5, and at cell 293 the two deviate in opposite directions (Im ω̃ +1.1e-11
vs −2.1e-11 at α = π, where |q_new/q_now| = 0.025). So the strips are where both SVDs round
off, and the θ = 0 gap is their difference (`weno_data/w4c_review/polish2_strips.py`).
icon4py's own θ = 30 field has the same strips: at CFL 0.5 the one-step ω of the strip cells
≥ 4 edge lengths from both seams deviates from that of the same-orientation cell 694 / 695
(normalised as above, max over the 59 α) by 1.5e-13 (median), that of the cells whose θ = 0
difference is below 3e-15 by 6.5e-15. So his θ = 0 cell sits in the worst strip and his θ = 30
cells where the pseudoinverses agree to round-off: one pair of implementations reproduces the
whole θ = 0 / θ = 30 contrast, and his θ = 0 and θ = 30 quadratic tables fit one build. The
strip pattern, a per-cell property of the pseudoinverse, supports the SVD attribution of the
θ = 0 gap; the direct test (injecting the capture's pseudoinverse into the icon4py granule,
Next item 4) is still open.

Rerun (from the workspace root; each job waits ≤ 1500 s for the lock):

```
sbatch --parsable weno_data/slurm/w4c_dispersion.sbatch gtfn_cpu 1500 paper - \
    "validation::theta30 and (paper or stability)"                 # gated sets, ~7 min
sbatch --parsable [--dependency=afterany:<id>] weno_data/slurm/w4c_dispersion.sbatch \
    gtfn_cpu 1500 full full "validation::theta30 and env"          # full tables, ~17 min
icon4py/.venv/bin/python weno_data/slurm/w4c_analysis.py          # numpy comparison tables
```

The `env` set resumes from `theta30_parts/full_<backend>/` and starts no CFL after
`ICON4PY_DISPERSION_DEADLINE` (the job sets it to its start + 26 min), so chained jobs
complete a list that does not fit one; `ICON4PY_DISPERSION_REUSE_PARTS=1` re-checks the gated
sets from their saved parts without the iteration.

## W6 — convergence order of the quadratic WENO schemes (torus patch, 2026-09-11)

The paper has no convergence study; it states that for smooth solutions the WENO
discretisation delivers identical results to the pure scheme (§2.3, §4). Measured here, with
the open question of `weno_idealized_scope.md` ("Open question for the order study").

**Setup.** `model/driver/tests/driver/scientific_validation/test_weno_order_study.py` with
`experiment_configs/weno_order_study_gaussian_2d.yaml`: the (38, 44, 400 m) torus family
(3344 / 13376 / 53504 cells for factors 1 / 2 / 4, 214016 / 856064 for the 8x / 16x
members; one 17.6 × 13.2 km domain), 3 identical levels, `gaussian_2d` at the domain centre with
`decay_radius 0.35`, constant diagonal wind, a quarter period, no limiter, each member at
the yaml's CFL 0.11 (124 / 249 / 499 / 999 / 1998 steps). The members stop short of the
quarter period: the step count is the floor of 0.25 s / Δt, and `RelativeTime` rounds Δt to
microseconds (2002 / 1001 / 500 / 250 / 125 µs, 0.1 % short from 4x on), so they end at
0.2482 / 0.2492 / 0.2495 / 0.24975 / 0.24975 s. On the coarsest member the e-folding
radius is 1753 m = 4.4 edge lengths = 5.1 cell rows; 139 cells lie inside it, 963 of 3344
inside the 1e-3 radius. Errors are relative L1 / L2 / Linf against the translated analytic
Gaussian, evaluated at the time the driver integrated to (steps × rounded Δt); the distance is
`‖q_row − q_3‖/‖q_3‖` from the saved final tracers. Differences from
`linear_advection_tests.py::test_horizontal_advection_convergence`: the family, the yaml
(`decay_radius` 0.35 instead of 0.25, 3 levels, quarter period, no limiter), the L2 norm and
the distance, and one time step per member (constant CFL) instead of the finest member's
step for all. The 132 row uses the optimised set for the type-VI assembly and unit weights
in its WENO branch at run time (f90 3684), as the Fortran does.

**Runs** (debug-partition jobs that own the pytest lock, one pytest per row; results merged
per (row, factor) into `weno_data/slurm/w6s_results_<backend>.json`, tracers next to it):
factors 1 / 2 / 4 on gtfn_cpu (compute node, 16 cores, `ulimit -t` unlimited; jobs 859589,
859590, 859598, 859640), 1 / 2 / 4 / 8 on dace_gpu (859600, 859641, 859682, 859683) and a
16x member (856064 cells, 1998 steps) for 3 / 103 OPT / 103 UNITY (859780).
Backend agreement was checked on the common members only, x1 / x2 / x4 of 3, 103 OPT,
103 UNITY and 132 (2 and 102 ran on gtfn_cpu only, the x8 and x16 members on dace_gpu only):
final tracers within 1.55e-15 (max |Δq|), relative errors within 1.72e-12. gtfn_cpu on the
compute node reproduces W6's login-node runs bit for bit (miura tracers; the errors of 2 / 3
/ 102 / 103 OPT / UNITY of the killed first attempt).

Relative errors (x1-x4 gtfn_cpu, x8/x16 dace_gpu) and wall time of the driver run in s (gtfn_cpu, warm cache / dace_gpu, including the per-grid-size build):

| row | x | L1 | L2 | Linf | wall gtfn_cpu | wall dace_gpu |
|---|---|---|---|---|---|---|
| 2 | 1 | 1.014e-02 | 8.471e-03 | 1.083e-02 | 22 | - |
|  | 2 | 2.109e-03 | 1.689e-03 | 1.960e-03 | 16 | - |
|  | 4 | 4.923e-04 | 3.873e-04 | 4.352e-04 | 21 | - |
| 3 | 1 | 7.331e-03 | 7.149e-03 | 1.154e-02 | 28 | 306 |
|  | 2 | 9.413e-04 | 9.283e-04 | 1.511e-03 | 19 | 273 |
|  | 4 | 1.183e-04 | 1.169e-04 | 1.906e-04 | 25 | 251 |
|  | 8 | 1.481e-05 | 1.464e-05 | 2.379e-05 | - | 278 |
|  | 16 | 1.853e-06 | 1.830e-06 | 2.971e-06 | - | 427 |
| 102 | 1 | 3.057e-02 | 4.600e-02 | 1.152e-01 | 21 | - |
|  | 2 | 5.967e-03 | 1.244e-02 | 4.883e-02 | 17 | - |
|  | 4 | 1.083e-03 | 3.013e-03 | 1.906e-02 | 21 | - |
| 103 OPT | 1 | 1.239e-03 | 1.299e-03 | 2.088e-03 | 36 | 80 |
|  | 2 | 2.932e-04 | 2.972e-04 | 4.382e-04 | 34 | 57 |
|  | 4 | 1.330e-04 | 1.292e-04 | 1.835e-04 | 91 | 76 |
|  | 8 | 6.533e-05 | 6.293e-05 | 8.883e-05 | - | 136 |
|  | 16 | 3.253e-05 | 3.127e-05 | 4.405e-05 | - | 340 |
| 103 UNITY | 1 | 5.730e-03 | 5.947e-03 | 9.658e-03 | 32 | 162 |
|  | 2 | 7.702e-04 | 7.885e-04 | 1.284e-03 | 34 | 21 |
|  | 4 | 1.138e-04 | 1.200e-04 | 1.937e-04 | 96 | 38 |
|  | 8 | 2.468e-05 | 2.643e-05 | 4.097e-05 | - | 96 |
|  | 16 | 9.176e-06 | 9.219e-06 | 1.351e-05 | - | 305 |
| 132 | 1 | 5.860e-03 | 6.139e-03 | 9.978e-03 | 34 | 71 |
|  | 2 | 8.505e-04 | 8.952e-04 | 1.453e-03 | 49 | 53 |
|  | 4 | 1.645e-04 | 1.774e-04 | 2.792e-04 | 112 | 68 |
|  | 8 | 5.547e-05 | 5.654e-05 | 9.337e-05 | - | 127 |

Slopes ± stderr (local rates between neighbouring members):

| row | factors | L1 | L2 | Linf |
|---|---|---|---|---|
| 2 | 1,2,4 | 2.18 ± 0.05 (2.27 / 2.10) | 2.23 ± 0.06 (2.33 / 2.12) | 2.32 ± 0.09 (2.47 / 2.17) |
| 3 | 1,2,4 | 2.98 ± 0.01 (2.96 / 2.99) | 2.97 ± 0.01 (2.95 / 2.99) | 2.96 ± 0.02 (2.93 / 2.99) |
| 3 | 1,2,4,8,16 | 2.99 ± 0.00 (2.96 / 2.99 / 3.00 / 3.00) | 2.99 ± 0.01 (2.95 / 2.99 / 3.00 / 3.00) | 2.98 ± 0.01 (2.93 / 2.99 / 3.00 / 3.00) |
| 102 | 1,2,4 | 2.41 ± 0.03 (2.36 / 2.46) | 1.97 ± 0.05 (1.89 / 2.05) | 1.30 ± 0.03 (1.24 / 1.36) |
| 103 OPT | 1,2,4 | 1.61 ± 0.27 (2.08 / 1.14) | 1.66 ± 0.27 (2.13 / 1.20) | 1.75 ± 0.29 (2.25 / 1.26) |
| 103 OPT | 1,2,4,8,16 | 1.27 ± 0.12 (2.08 / 1.14 / 1.03 / 1.01) | 1.30 ± 0.13 (2.13 / 1.20 / 1.04 / 1.01) | 1.34 ± 0.14 (2.25 / 1.26 / 1.05 / 1.01) |
| 103 UNITY | 1,2,4 | 2.83 ± 0.04 (2.90 / 2.76) | 2.82 ± 0.06 (2.91 / 2.72) | 2.82 ± 0.05 (2.91 / 2.73) |
| 103 UNITY | 1,2,4,8,16 | 2.35 ± 0.17 (2.90 / 2.76 / 2.20 / 1.43) | 2.36 ± 0.16 (2.91 / 2.72 / 2.18 / 1.52) | 2.39 ± 0.15 (2.91 / 2.73 / 2.24 / 1.60) |
| 132 | 1,2,4 | 2.58 ± 0.12 (2.78 / 2.37) | 2.56 ± 0.13 (2.78 / 2.34) | 2.58 ± 0.12 (2.78 / 2.38) |
| 132 | 1,2,4,8 | 2.25 ± 0.19 (2.78 / 2.37 / 1.57) | 2.26 ± 0.18 (2.78 / 2.34 / 1.65) | 2.26 ± 0.19 (2.78 / 2.38 / 1.58) |

Distance to scheme 3, `‖q − q_3‖/‖q_3‖`, and its ratio to scheme 3's own error (same norm):

| row | x | L2 | Linf | L2 ÷ err₃ | Linf ÷ err₃ |
|---|---|---|---|---|---|
| 2 | 1 | 6.002e-03 | 8.012e-03 | 0.84 | 0.69 |
|  | 2 | 1.503e-03 | 1.893e-03 | 1.62 | 1.25 |
|  | 4 | 3.751e-04 | 4.428e-04 | 3.21 | 2.32 |
| 102 | 1 | 4.259e-02 | 1.053e-01 | 5.96 | 9.12 |
|  | 2 | 1.221e-02 | 4.739e-02 | 13.15 | 31.35 |
|  | 4 | 2.998e-03 | 1.887e-02 | 25.65 | 99.01 |
| 103 OPT | 1 | 5.997e-03 | 1.000e-02 | 0.84 | 0.87 |
|  | 2 | 6.727e-04 | 1.087e-03 | 0.72 | 0.72 |
|  | 4 | 6.290e-05 | 4.810e-05 | 0.54 | 0.25 |
|  | 8 | 5.078e-05 | 6.504e-05 | 3.47 | 2.73 |
|  | 16 | 2.971e-05 | 4.108e-05 | 16.23 | 13.83 |
| 103 UNITY | 1 | 1.294e-03 | 1.904e-03 | 0.18 | 0.17 |
|  | 2 | 1.490e-04 | 2.353e-04 | 0.16 | 0.16 |
|  | 4 | 1.674e-05 | 1.433e-05 | 0.14 | 0.08 |
|  | 8 | 1.325e-05 | 1.719e-05 | 0.91 | 0.72 |
|  | 16 | 7.612e-06 | 1.054e-05 | 4.16 | 3.55 |
| 132 | 1 | 1.130e-03 | 1.601e-03 | 0.16 | 0.14 |
|  | 2 | 1.073e-04 | 9.946e-05 | 0.12 | 0.07 |
|  | 4 | 7.171e-05 | 8.932e-05 | 0.61 | 0.47 |
|  | 8 | 4.365e-05 | 7.002e-05 | 2.98 | 2.94 |

Slopes of the distance (L2 / Linf, local rates): 2: 2.00 ± 0.00 / 2.09 ± 0.00; 102:
1.91 ± 0.06 / 1.24 ± 0.05.

| row | factors | L2 | Linf |
|---|---|---|---|
| 103 OPT | 1,2,4 | 3.29 ± 0.08 (3.16 / 3.42) | 3.85 ± 0.37 (3.20 / 4.50) |
| 103 OPT | 1,2,4,8,16 | 1.90 ± 0.41 (3.16 / 3.42 / 0.31 / 0.77) | 1.99 ± 0.56 (3.20 / 4.50 / −0.44 / 0.66) |
| 103 UNITY | 1,2,4 | 3.14 ± 0.01 (3.12 / 3.15) | 3.53 ± 0.29 (3.02 / 4.04) |
| 103 UNITY | 1,2,4,8,16 | 1.83 ± 0.39 (3.12 / 3.15 / 0.34 / 0.80) | 1.88 ± 0.49 (3.02 / 4.04 / −0.26 / 0.71) |
| 132 | 1,2,4 | 1.99 ± 0.81 (3.40 / 0.58) | 2.08 ± 1.11 (4.01 / 0.16) |
| 132 | 1,2,4,8 | 1.47 ± 0.47 (3.40 / 0.58 / 0.72) | 1.37 ± 0.65 (4.01 / 0.16 / 0.35) |

Hybrid 132, fraction of (cell, level) points whose selection mask **of the last time step**
(computed from the tracer at its start, i.e. essentially the final state) took the WENO
branch: 0.9964 / 0.9818 / 0.9646 / 0.9475 for x1 / x2 / x4 / x8. The threshold
`5e-5·(q + 1e-10)²` is relative, so the far field (q → 0) is always WENO; the plain cells
are the Gaussian's core and their share grows with resolution (0.4 % → 5.3 %).

**Verdict on "for smooth solutions the WENO discretisation delivers identical results to
the pure scheme".** Not supported, neither at practical resolution nor asymptotically. At
the coarse members the 103 rows differ from scheme 3 by a sizeable fraction of scheme 3's
own error (UNITY 18 / 16 / 14 % in L2 at x1 / x2 / x4; OPTIMIZED 84 / 72 / 54 %, and
OPTIMIZED is then 5.5x more accurate than 3 on the coarsest member); the distance first
shrinks at rate 3.1-3.3, slightly faster than the errors, which is what the claim would
predict. From the 4x to the 8x member on the distance stops converging at third order and
turns towards first order (local rates 0.31 → 0.77 OPT, 0.34 → 0.80 UNITY), and the
schemes' own errors follow: 103 OPTIMIZED is first order below 200 m edge length (local
rates 1.14, 1.03, 1.01; 17.6x the error of 3 at x16), 103 UNITY drops from 2.9 to 1.43
(5x the error of 3 at x16), the hybrid to 1.57 at x8. Scheme 3 itself stays at 3.00 to the
16x member, and the 3 / 103 tracers agree across gtfn_cpu and dace_gpu to 1.55e-15 on the
x1-x4 members (x8 and x16 ran on dace_gpu only), so this is the scheme, not the grid, the
time step or a backend. Only the direction of the claim holds on the coarse members: UNITY
up to x4 (≤ 18 % of scheme 3's error) and the hybrid up to x2 (≤ 16 %) are close to 3,
OPTIMIZED is not.

**Mechanism (a finding for Andreas; the port matches the Fortran per step to 3.5e-9, W5, so
this is the Fortran's behaviour).** Checked at coefficient level by the W6 review (δ =
−1.6213e-3 on the study's own 25 m grid) and pinned by the numpy unit test
`tests/tracer_advection/unit_tests/test_weno_type_vi_bias.py` (the port's coefficient
functions on a torus patch at edge lengths 1 and 1/2: the type-VI candidates return (1 − S)
times the derivatives of a quadratic to 3.5e-14, asserted to 1e-8; δ on a Gaussian core
−1.6132e-3 / −1.6191e-3 OPTIMIZED and −4.1627e-4 / −4.1638e-4 UNITY at h/r_e = 1/6 / 1/12,
asserted within 1 % of −1.6214e-3 / −4.1647e-4 and of each other, their h²-extrapolation
−1.62107e-3 / −4.16409e-4 within 0.1 % of the closed form; the test fails if the assembly is
changed, on purpose). Fortran-side confirmation (W6-F, reviewed and accepted):
icon-ajocksch commit `3b86566381`, `CAPTURE_NOTES.md` "WENO dispersion (W6-F)": in his Fortran
the one-step α² damping coefficient A of 103 at CFL 0.01 is 7.83e-4 (optimised) / 1.98e-4 (all
ones) against ~1e-8 for scheme 3. The damping is a property of 103, not of the sampled cell: a
numpy replica that reproduces our Fortran build's 103 table on his grid to 4.3e-13 gives A > 0
at every cell, the phase average A = −(δ/2)(1 − 2 CFL) to 0.1 % (optimised) / 0.3 % (ones)
over CFL 0.01-0.4, and an optimised / ones ratio 3.903 against δ's 3.898. The single-cell rows
with Im ω̃ < 0 in that table are a phase artefact (the phase-averaged one-step Im ω̃ is
positive in all rows). Growth over many steps is a separate matter: the numpy replica and the
Fortran (W6-G, `icon-ajocksch` commit `96e64fe27a`, `CAPTURE_NOTES.md` "WENO multi-step
stability (W6-G)") show a multi-step grid-scale instability of 103 at θ = 0; limits and cause
are under review. The type-VI candidates are assembled as
`A⁺_full − Σ_{i∈group} d_i A⁺_i` (`weno_least_squares.compute_weno_pseudoinverse_quadratic`,
f90 2670-2680). For smooth data every fitted candidate reproduces the derivatives, so a
type-VI candidate returns `(1 − S)` times them, with `S` the group's weight sum
(OPTIMIZED: 2 × 2.9915 = 5.983; UNITY: 8). Its smoothness indicator is quadratic in the
coefficients (`accumulate_weno_candidate_flux_weights.py`, f90 2996-3008), hence
`(1 − S)²` = 24.8 / 49 times that of the fitted candidates, and its weight
`d/(β + ε)²` is `(1 − S)⁴` = 617 / 2401 times smaller than the linear weights assume — for
every resolution, so the nonlinear weights never tend to the linear ones. The normalised
blend then returns `(1 + δ)` times the true derivatives with
`δ = (D + 3(1 − S)⁻³)/(D + 3(1 − S)⁻⁴) − 1` (`D` = sum of the fitted candidates' weights,
17.95 / 24): δ = −1.62e-3 (OPTIMIZED), −4.16e-4 (UNITY). A gradient that is short by a
constant fraction is a first-order numerical diffusion `∝ |δ|·h`: the WENO rows' peak is
lower than scheme 3's (x16: −4.1e-5 OPT, −1.05e-5 UNITY), and the predicted UNITY / OPT
ratio of that term, 0.257, is what the distances show at x8 (0.261 L2, 0.264 Linf) and x16
(0.256 L2, 0.257 Linf). (With exactly linear weights the same assembly would return
`3/Σ_j d_j` of the derivatives, 1/9 for UNITY and 0.14 for OPTIMIZED — the scope note's
open question; on smooth data the weights never reach that limit, which is what keeps the
scheme consistent up to δ.) The hybrid assembles
with OPTIMIZED and blends with unit weights (δ = −1.21e-3) outside its plain core.

**Gates** (`_ROWS`, measured values in comments). 3: 3 ± 0.1 on the (1, 2, 4) fit and on
the local rate between the last two members in the results file. 2 and 102: regression
guards on the (1, 2, 4) fit at `linear_advection_tests._measured` width (± 0.5) around the
gtfn_cpu slopes. 103 OPT, 103 UNITY and 132 document the deficiency of the type-VI
construction above rather than an order of accuracy: they are gated on the local rate
between the last two members in the results file, ± 0.1 around the rate measured for that
member pair (x2-x4, x4-x8, x8-x16), every band below the third-order one, so the gate tells
first from third order (it catches a return to third order or a shifted transition, not the
value of δ); and on the ratio of the last member's L2 error to scheme 3's at the same factor,
± 5 % around the measured one (x4 / x8 / x16: OPT 1.105 / 4.299 / 17.09, UNITY 1.027 / 1.805 /
5.038; 132 1.518 / 3.862 at x4 / x8), which on the finest members follows the size of the δ
diffusion (a 6 % larger x16 OPT error fails it). δ itself is pinned by the unit test. Their
(1, 2, 4) fit is printed only. `ICON4PY_WENO_ORDER_STUDY_CHECK_ONLY=1`
passes on both results files without a rerun (gtfn_cpu all rows, last pair x2-x4; dace_gpu
with `ICON4PY_WENO_ORDER_STUDY_ROWS=miura3,miura3_weno_opt,miura3_weno_unity,miura3_weno_hybrid`,
last pair x8-x16, x4-x8 for 132; logs `weno_data/slurm/w6fix_checkonly_<backend>.log`, with
the ratio gate `weno_data/slurm/polish_checkonly_<backend>.log`).

**Rerun** (santis, husk; one debug job per row, chained; each job takes and releases
`weno_data/pytest.lock` itself, so it survives the submitting agent). Workspace script
`weno_data/slurm/w6s_row.sbatch <backend> <factors> <lock wait s> <row> [<row> ...]`
(one pytest per row argument; a comma-separated argument runs those rows in one pytest,
which generates the grids once — used for the 16x member):

```
cd <workspace>    # husk confines --output below the submitting directory
id=$(sbatch --parsable weno_data/slurm/w6s_row.sbatch gtfn_cpu 1,2,4 600 miura3 miura_weno miura)
id=$(sbatch --parsable --dependency=afterany:$id weno_data/slurm/w6s_row.sbatch gtfn_cpu 1,2,4 900 miura3_weno_opt)
id=$(sbatch --parsable --gpus=1 --dependency=afterany:$id weno_data/slurm/w6s_row.sbatch dace_gpu 1,2,4,8 900 miura3)
id=$(sbatch --parsable --gpus=1 --mem=256G --dependency=afterany:$id weno_data/slurm/w6s_row.sbatch dace_gpu 16 600 miura3,miura3_weno_opt,miura3_weno_unity)
until ! squeue -j $id -h | grep -q .; do sleep 60; done
```

Inside, the job sets the environment of `docs/run_jocksch_reference_gpu.sbatch` plus
`ICON4PY_WENO_ORDER_STUDY_RESULTS=weno_data/slurm/w6s_results_<backend>.json`,
`ICON4PY_WENO_ORDER_STUDY_FACTORS` and `ICON4PY_WENO_ORDER_STUDY_ROWS`, and runs
`pytest -n0 --backend=<backend> --level=validation model/driver/tests/driver/scientific_validation/test_weno_order_study.py`;
success is pytest's "passed" line in `weno_data/slurm/w6s_<row>_<backend>.log` (also
collected in `w6s_exit_codes`). Cost: gtfn_cpu 1.5-7 min per row (1 / 2 / 4); dace_gpu builds
one variant of every program per grid size (4-5 min per new size for the driver's
factories, about 1 min for the 103 stencils); once built, 0.5-2.5 min per member up to 8x,
5-7 min per 16x member including its build. Check a
results file without running: the same pytest with
`ICON4PY_WENO_ORDER_STUDY_CHECK_ONLY=1` (and `ICON4PY_WENO_ORDER_STUDY_ROWS` for a subset).
Tables: `weno_data/slurm/w6s_analysis.py <gtfn_cpu json> <dace_gpu json>` (numpy).

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
3. W5 / W6 follow-ups: gates on all backends (`dace_gpu` > `dace_cpu` > `gtfn_gpu`). The 103
   order study is done (W6, section above), and so is the Fortran confirmation of the
   multi-step instability seen in the numpy replica of 103 (W6-G, committed on the Fortran
   tree, `icon-ajocksch` `96e64fe27a`; its review is running). Open: the pseudoinverse
   injection test (item 4); the 103 launch-count restructuring (27 candidates × 2 launches)
   before any performance claim.
4. Dispersion relation: done at θ = 0 (W4a, W4b capture) and θ = 30° (W4c, section
   above). Open: the provenance of his `dispersion_ffsl_{0,30}[_inexact].txt` (none of
   schemes 2, 3, 5; question for Andreas), and the direct test that the θ = 0
   icon4py-vs-build gap of scheme 3 is the SVD pseudoinverse (inject the capture's
   `lsq-coefficients` pseudoinverse into the icon4py granule and rerun θ = 0); the per-cell
   strip pattern (W4c section) already supports that attribution.
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
