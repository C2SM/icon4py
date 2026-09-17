# Reproduce the dycore optimisation measurements

[Read the analysis first](DYCORE_GRANULE_ANALYSIS.md). Run the commands below **from the `dycore-optimizations` checkout**.
The small launcher creates a detached, private checkout of the pinned experiment
inside the job output directory and runs the recorded harness there. Your current
branch and source files stay unchanged. This replays the measured implementations;
the solver implementation matches this branch, which is based on C2SM's
`mi300_opt` at `397d774a1`. The replay deliberately does not incorporate later
uncommitted edits to the review checkout.

**Reproduction still depends on the `mi300_opt` experiment archive.** This
branch contains the optimisations and launchers; the complete benchmark harness
and original evidence are stored on `dganellari/icon4py:mi300_opt`. The launcher
runs the pinned archive commit `cdc034acb` in a private checkout, rather than
benchmarking the current review working tree.

Stay on `dycore-optimizations`. If the archive commit is missing locally, download
it before submitting (this works regardless of your remote names):

```bash
git fetch https://github.com/dganellari/icon4py.git mi300_opt
```

`git fetch` downloads the archive history; it does not switch branches or modify
checked-out source files. No checkout of `mi300_opt` is required. Launch the jobs
below from `dycore-optimizations` after fetching.

**There is not yet a measured combined 5% result to reproduce.** We measured
2.13% less device time from theta compiler fusion and 3.28% additional reduction
from solver fusion. Their product suggests 5.34%; the direct combined experiment
below is needed to confirm or revise it. Do not sum the percentages.

## Enable theta fusion in normal dycore runs

With the GT4Py shared-output fusion patch installed, normal model configuration
can enable the same restricted callback used in the measured experiment:

```bash
export ICON4PY_DACE_THETA_FUSION=1  # Enable before constructing the dycore/backend.
# Set to 0, or leave unset, to disable the model's opt-in.
```

`model_options.py` registers `TopLevelDataFlowVerticalSplitCallBack` only for
`compute_rho_theta_pgrad_and_update_vn`. The callback selects the theta output,
matching horizontal ranges and different vertical bands; other candidates keep
shared-output splitting disabled. An unpatched compiler or an existing conflicting
callback produces a clear error. Configure each comparison in a fresh process
and use separate persistent cache directories to retain the generated SDFGs.
This enables graph-based matching across specializations; it does not claim
validation or a speedup for every grid/IAU variant.

This switch configures normal runs of this branch. The pinned replay commands
below retain their original benchmark-controlled A/B configuration and do not
exercise later model-options edits. Solver fusion remains applied in the model.

## Experimental solver compiler option

The new [scan-input fusion patch](../../patches/gt4py-scan-input-fusion.patch)
moves coefficient calculations into a scan in the compiler, so scientists can
keep the original Python equations. It is a separate, CPU-validated candidate;
the measured 3.28% solver gain above belongs to the existing Python rewrite.

For development with the GT4Py review branch, install both compiler changes
(the second patch applies after the shared-output fusion commit `857e718d`),
then set this before constructing the model/backend:

```bash
export ICON4PY_DACE_SOLVER_FUSION=1
# Set to 0, or leave unset, to disable this compiler pass.
```

Normal `model_options.py` sets `optimization_args["fuse_scan_inputs"]` only for
the predictor and corrector vertical solver programs. GT4Py consumes this option
in its DaCe translator **before building the SDFG**. The remaining options then
reach `gt_auto_optimize`. Theta's callback, in contrast, runs during SDFG
optimization. Both options are explicit and default off.

**This branch still contains the measured Python solver rewrite.** Turning the
compiler option off does not undo that rewrite, and turning it on does not add
a demonstrated further speedup. To validate the compiler as its replacement,
compare the original solver equations with this option off/on in a separate
validation checkout; first require full-granule GPU correctness, then timing.
The pinned replay below does not exercise this new pass.

Local checks cover 11 compiler tests and 14 model-option tests. A separate
compiled CPU comparison using the original ICON forward sweep matched both the
original and manually fused results exactly for three randomized inputs, while
removing four two-dimensional coefficient arrays. This standalone count is not
the measured full solver's net three-array reduction. GPU validation, mixed
precision and full-granule performance of the compiler replacement remain open.

## Source and environment

The launcher uses experiment commit **`cdc034acb`**, including both measured
snapshots and the new combined benchmark. It checks installed GT4Py/DaCe revisions,
sets Python imports to the private model copy, and records `REPLAY.json` next to
the results. No branch switch or installed-source patch is performed.

The measured stack is pinned to:

| Repository | Revision |
|---|---|
| Icon4Py measured model base | `397d774a17135702b411d97edd4fb42cd0e21566` |
| Icon4Py experiment bundle, including grid-level fix | `cdc034acb` |
| GT4Py | `eb763b97515a76c70e60befcba44dcf3bd18fb65` |
| DaCe | `5115128a73dc518071dbe9580b63d382540efe46` |

In the private snapshot, `amd_scripts/review_2026_09_16/BASES.json` records the original
bases, and each experiment's `PATCH_MANIFEST.json` checks its compiler source.
The GT4Py scalar-conversion prerequisite is
`amd_scripts/review_2026_09_16/patches/02-domain-scalar-gt4py.patch`; apply it in
the pinned GT4Py checkout if absent. The wrappers apply the experimental fusion
module to a **private GT4Py copy**; do not patch the installed compiler with the
review patch manually when reproducing the original timings; the harness
installs the matching compiler module into its private overlay.

The cluster jobs require the existing ICON regional/grid input data and the
configured `venv_mi300` or `venv_gh200`, with GT4Py and DaCe installed from
the pinned editable checkouts. The launcher links those dependencies beside the
private model and ensures that the private model is imported. Follow the repository's
environment/data setup for a fresh checkout; the wrappers do not provision the
input data or those environments. If the review checkout has a `testdata`
directory or symlink, it is also linked into the private checkout. They select the recorded CSCS uenvs and
serial compilation settings. These are cluster reproduction instructions, not
a claim that a fresh CPU-only clone can reproduce GPU timings.

The explicit grid-level fix is included in the experiment commit. Every run
must report **regional with 120 levels**. Do not edit or transfer source files
into a checkout while its jobs are running. The user submits/manages all jobs.

## Run the separate comparisons on MI300A

From `dycore-optimizations` on Beverin:

```bash
sbatch --export=ALL,OPTIMIZATION_COMPARISON=theta amd_scripts/benchmark_optimizations_amd.sh
sbatch --export=ALL,OPTIMIZATION_COMPARISON=solver-increment amd_scripts/benchmark_optimizations_amd.sh
```

These are separate allocations with separate output directories:

| Experiment | A | B | Output |
|---|---|---|---|
| Theta primary comparison | Original code | Compiler theta fusion | `results/compiler_vs_native/` |
| Solver incremental comparison | Compiler theta fusion | Same theta + both fused solvers | `results/solver_coefficients_in_scan/` |

The launcher selects only the theta-versus-native comparison, omitting the
historical Python-rewrite comparison. The solver experiment measures the
**increment on top of theta**, not solver-only versus original. Both separate
recipes are AMD-only; the launcher rejects them on NVIDIA.

## Measure the complete combination directly

On Beverin:

```bash
sbatch --export=ALL,OPTIMIZATION_COMPARISON=combined amd_scripts/benchmark_optimizations_amd.sh
```

On Santis, also from `dycore-optimizations`, with its configured GH200 environment:

```bash
sbatch --export=ALL,OPTIMIZATION_COMPARISON=combined amd_scripts/benchmark_optimizations_nvidia.sh
```

Both use original code as A and compiler theta fusion plus both fused solvers
as B. All results go to `amd_scripts/optimization_runs/amd_<job>/results/` or
`amd_scripts/optimization_runs/nvidia_<job>/results/`. The adjacent `snapshot/`
directory retains the exact model and build outputs for inspection. This is a prepared experiment;
there are no completed combined results yet. Each GPU retains its native launch
configuration. No profilers run during these timings.

## Accept and compare results

Each experiment uses 12 balanced A/B quartets and interleaved identical-arm
controls. Require `COMPLETE`, passing numerical/code/source checks, and an actual
120-level grid. Inspect `TIMING_SUMMARY.md` / `.json`, the raw `timing.*.json`,
per-program `timers.timing.*.json`, and `code_audit.json`; a finished Slurm job
alone is not evidence that validation passed.

For each GPU, combined device reduction is `100 × (1 − B/A)` using
`granule_device.baseline_ms` and `variant_ms` under the summary's `metrics`.
Report `granule_wall` separately. Review raw and control-adjusted confidence
intervals and order effects; the earlier conservative screen additionally
requires the mean saving to exceed `|mean A/A contrast| + 2 × SD(A/A contrasts)`.
A new node need not reproduce the exact old milliseconds or percentage.

To compare vendors, first verify matching grid/input fingerprints and compatible
source/dependency manifests. Report `AMD_A / GH200_A` before and
`AMD_B / GH200_B` after, with uncertainty. A faster AMD run alone does not prove
that the vendor gap shrank; GH200 may also improve.

The historical global/regional comparison and previous GPU validation remain
in the analysis. These optimisation recipes intentionally measure regional/120;
they do not establish global-grid safety or speedup.

To verify the retained original evidence without a GPU, from this branch
after a replay has prepared its snapshot:

```bash
python3 amd_scripts/optimization_runs/amd_<job>/snapshot/icon4py/amd_scripts/review_2026_09_16/verify_evidence.py
```

That reconstructs the separate measured gains and checks the archived evidence.
It does not replace running the combined experiment.
