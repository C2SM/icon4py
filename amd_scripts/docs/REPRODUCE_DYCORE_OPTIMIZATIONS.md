# Reproduce the dycore optimisation measurements

[Read the analysis first](DYCORE_GRANULE_ANALYSIS.md). The small
`dycore-optimizations` branch is for code review. Exact measured source snapshots,
job wrappers and raw evidence live on **`mi300_opt`**, so they do not expand this
PR into a benchmark-framework review. Keep both branches available in the fork.

**There is not yet a measured combined 5% result to reproduce.** We measured
2.13% less device time from theta compiler fusion and 3.28% additional reduction
from solver fusion. Their product suggests 5.34%; the direct combined experiment
below is needed to confirm or revise it. Do not sum the percentages.

## Source and environment

Use an isolated, configured benchmark checkout of `mi300_opt` at commit
**`cdc034acb`**. This includes both measured snapshots and the new combined
benchmark. Do not run these wrappers from the current-main optimisation branch:
the measured implementation and dependency versions must match.

The measured stack is pinned to:

| Repository | Revision |
|---|---|
| Icon4Py measured model base | `397d774a17135702b411d97edd4fb42cd0e21566` |
| Icon4Py experiment bundle, including grid-level fix | `cdc034acb` |
| GT4Py | `eb763b97515a76c70e60befcba44dcf3bd18fb65` |
| DaCe | `5115128a73dc518071dbe9580b63d382540efe46` |

In that checkout, `amd_scripts/review_2026_09_16/BASES.json` records the original
bases, and each experiment's `PATCH_MANIFEST.json` checks its compiler source.
The GT4Py scalar-conversion prerequisite is
`amd_scripts/review_2026_09_16/patches/02-domain-scalar-gt4py.patch`; apply it in
the pinned GT4Py checkout if absent. The wrappers apply the experimental fusion
module to a **private GT4Py copy**; do not patch the installed compiler with the
new-main review patch when reproducing the original timings.

The cluster jobs require the existing ICON regional/grid input data and the
configured `venv_mi300` or `venv_gh200`, with editable packages pointing at this
benchmark checkout and its sibling GT4Py/DaCe checkouts. Follow the repository's
environment/data setup for a fresh checkout; the wrappers do not provision the
input data or those environments. They select the recorded CSCS uenvs and
serial compilation settings. These are cluster reproduction instructions, not
a claim that a fresh CPU-only clone can reproduce GPU timings.

The explicit grid-level fix is included in the experiment commit. Every run
must report **regional with 120 levels**. Do not edit or transfer source files
into a checkout while its jobs are running. The user submits/manages all jobs.

## Run the separate comparisons on MI300A

From that benchmark checkout on Beverin:

```bash
sbatch amd_scripts/theta_shared_timing/run_amd.sh
sbatch amd_scripts/solver_scan_fusion/run_amd.sh
```

These are separate allocations with separate output directories:

| Experiment | A | B | Output |
|---|---|---|---|
| Theta primary comparison | Original code | Compiler theta fusion | `amd_scripts/theta_shared_timing_runs/amd_<job>/compiler_vs_native/` |
| Solver incremental comparison | Compiler theta fusion | Same theta + both fused solvers | `amd_scripts/solver_scan_fusion_runs/amd_<job>/solver_coefficients_in_scan/` |

The theta wrapper also runs its historical comparison against the Python theta
rewrite. That extra comparison is not another optimisation to add. The solver
experiment measures the **increment on top of theta**, not solver-only versus
original. Both separate recipes are MI300A recipes; do not use them on GH200.

## Measure the complete combination directly

On Beverin:

```bash
sbatch amd_scripts/combined_fusion/run_amd.sh
```

On Santis, from its equivalently configured benchmark checkout:

```bash
sbatch amd_scripts/combined_fusion/run_nvidia.sh
```

Both use original code as A and compiler theta fusion plus both fused solvers
as B. Results go to `amd_scripts/combined_fusion_runs/amd_<job>/` and
`amd_scripts/combined_fusion_runs/nvidia_<job>/`. This is a prepared experiment;
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

To verify the retained original evidence locally without a GPU, from the
`mi300_opt` experiment checkout:

```bash
python3 amd_scripts/review_2026_09_16/verify_evidence.py
```

That reconstructs the separate measured gains and checks the archived evidence.
It does not replace running the combined experiment.
