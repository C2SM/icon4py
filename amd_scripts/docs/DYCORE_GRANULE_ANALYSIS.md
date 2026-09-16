# Dycore performance: regional and global grids

MI300A was close to GH200 on the global mesh, but substantially slower on the
operational regional mesh. We investigated that difference and found two code
changes that improve the regional calculation on MI300A.

[Reproduction instructions](REPRODUCE_DYCORE_OPTIMIZATIONS.md) cover each
measured increment and the pending direct combined comparison.

All results below concern the **solve_nonhydro granule**, not diffusion or a
complete model timestep. Both meshes use 120 vertical levels. Device time is
the sum of the GPU time recorded for the programs inside the granule.

## Starting point: the grid changes the comparison

| Mesh | Cells (approximately) | MI300A device time | GH200 device time | MI300A / GH200 |
|---|---:|---:|---:|---:|
| Global, R02B06 | 327,000 | 35.2163 ms | 31.8123 ms | **1.107×** |
| Regional, MeteoSwiss operational domain | 44,500 | 5.7351 ms | 3.8774 ms | **1.479×** |

Thus MI300A takes about 11% more device time on global and 48% more on regional.
For regional, the whole-call host-wall ratio is smaller: **1.339×**
(6.4133 versus 4.7893 ms). Device and wall results answer different questions;
the 48% figure is not an end-to-end forecast for the weather model.

Regional is roughly one seventh the size, but also has different connectivity,
boundaries and active computation domains. It is not simply a smaller copy of
global. Cache measurements show more hits on regional on both chips, with a
larger improvement on GH200 under its counters. That does not prove that the
entire timing gap comes from cache capacity: AMD and NVIDIA counters count
different units, and AMD traffic leaving L2 can be served by MALL rather than HBM.
Regional's poorer spatial locality and different kernel structure remain relevant.
We have not assigned the gap to one exclusive cause.

## Two successful changes

**Theta-rho compiler fusion.** The original program produces fields across a
vertical column, then consumes them in separate vertical bands. The GT4Py change
allows the compiler to split that producer safely, retain its externally needed
outputs, and fuse compatible calculations using the existing DaCe transformations.
The measured regional program goes from six GPU kernels to five. This is an
opt-in GT4Py change; its default remains off.

**Vertical solver fusion.** Compute the tridiagonal coefficients inside the
forward sweep, immediately before they are used, instead of materialising their
full-column arrays first. Both predictor and corrector use this sweep. In the
measured full solver, each specialization loses one kernel and a net three
coefficient-sized intermediate arrays. This is a storage/code-generation result;
we did not measure the resulting HBM traffic reduction.

| Paired comparison, MI300A regional/120 | Granule device time | Reduction | Target-program reduction |
|---|---:|---:|---:|
| Original → compiler theta fusion | 5.511207 → 5.393807 ms | **2.13%** | Theta-rho **14.33%** |
| Compiler theta fusion → theta + solver fusion | 5.519307 → 5.338015 ms | **3.28% additional** | Both solvers **8.33%** |

The solver change also reduces whole-call wall time by **2.93%**. Each row is
its own same-node paired experiment; the rows ran on different nodes, so their
absolute baselines differ. Multiplying the measured reductions suggests about
**5.34% less device time combined**, but that is an estimate, not a direct
original-versus-combined measurement. The earlier Python theta rewrite is an
alternative to compiler fusion, not another gain to add.

## Evidence and remaining validation

The measured runs are jobs **639200** (theta, nid002952) and **639284** (solvers,
nid002926). Each used 12 balanced A/B quartets, interleaved identical-arm controls,
restored input state and source checksums. Both reported gains clear the
conservative control-noise screen. Validation covered 148 state arrays with
zero observed finite-value error and matching nonfinite patterns.

These timings belong to the preserved `mi300_opt` experiment revision. This
small Icon4Py branch ports the solver arithmetic to current upstream's staggered
vertical-level interface; its GPU granule validation and timing remain pending.
The port passes isolated embedded and compiled CPU comparisons against the
existing independent NumPy reference at 2, 40 and 120 levels in double precision.
Normal Icon4Py pytest collection was blocked locally by missing Serialbox;
mixed-precision collection also encounters an existing return-type mismatch in
the unchanged standalone scan. Neither check is claimed as passing.

The [GT4Py patch](../../patches/gt4py-shared-output-fusion.patch) contains the compiler
transformation and regression tests (22 focused tests and pre-commit checks pass).
GT4Py is a separate repository, so the change is carried here as an applyable
patch rather than copying the compiler into Icon4Py. Its base is GT4Py
`a461b874` (upstream main); the modified transformation source is
unchanged from the measured prototype. It does not automatically enable theta
fusion in Icon4Py. This branch contains the solver change, depth-parameterised
tests, compiler patch, this analysis and small launchers for reproducing the
experiments directly from this branch.
Benchmark scripts, raw data and unsuccessful experiments stay on the experiment
branch; the retained evidence is under `amd_scripts/review_2026_09_16/` there.

Direct combined timing, GH200 validation of these exact changes, and global-grid
correctness/performance checks are still needed before recommending both
optimisations for general use. No combined saving or reduction of the vendor
gap is claimed as measured yet.

## Where to review the code

- [Solver implementation](../../model/atmosphere/dycore/src/icon4py/model/atmosphere/dycore/stencils/solve_tridiagonal_matrix_for_w_forward_sweep.py):
  `_coefficient_forward_scan` computes coefficients as scalar values within each
  column's recurrence. The field operator supplies neighbouring-level inputs.
  Casts and arithmetic order are retained; the existing standalone scan API stays available.
- [Solver tests](../../model/atmosphere/dycore/tests/dycore/stencil_tests/test_solve_tridiagonal_matrix_for_w_forward_sweep.py):
  the independent NumPy recurrence now covers 2, 40 and 120 levels.
- [Compiler patch and tests](../../patches/gt4py-shared-output-fusion.patch):
  `allow_shared_data=False` in GT4Py's DaCe transformation layer; guarded splitting
  preserves external outputs and rejects unsupported aliasing, shifted/overlapping
  accesses and reductions. Existing DaCe fusion joins the compatible pieces.
  There is no DaCe core patch.

The measured theta selection used the existing optimizer callback
`TopLevelDataFlowVerticalSplitCallBack`. It opted in only for
`theta_v_at_edges_on_model_levels`, matching horizontal ranges and differing
vertical bands, and reset the flag for every candidate. It was not enabled
indiscriminately for other programs. Production integration must retain that
restriction until broader validation is complete.

The explicit benchmark-level fix and the GPU scalar conversion needed by the
profiling harness remain recorded in the experiment branch. They are not speedup
claims and are not bundled into this optimisation diff. Any reproduction must
verify that the actual grid has 120 levels rather than relying on the CLI label.
