# Dycore compiler fusion: results and reproduction

Both opt-in GT4Py passes improve the `solve_nonhydro` granule on regional and
global meshes. The latest paired runs reproduce the approximate compute gains
reported earlier. These results cover **120 levels, one GPU, and solve_nonhydro
only**; they exclude diffusion and the rest of a model timestep.

## What the times mean

- **Compute**, previously called GPU/device time here: the sum of timings inside
  the compiled GT4Py/DaCe programs. This includes their launches and
  synchronization, not just pure GPU kernel-event durations.
- **Granule wall**: elapsed time for the whole synchronized solve_nonhydro call,
  including Python dispatch, runtime overhead and work outside the program timers.

The metric scopes did not change. All percentages below are time reductions:
`100 * (original - optimized) / original`.

## Global versus regional: combined compiler benefit

Earlier regional: AMD 641726 / GH200 873329. Earlier global: AMD 644950 / GH200
875596\. Latest: **AMD 647055 (nid002424)** and **GH200 879091 (nid005260)**; both
jobs measured both meshes on their respective nodes. A global-only confirmation,
**GH200 879318**, repeated the same protocol on nid005260; its values are the
latest GH200 global entries below. Regional has 44,528 cells; global has 327,680.

| GPU / mesh      | Earlier compute reduction | Latest compute reduction | Earlier wall reduction | Latest wall reduction |
| --------------- | ------------------------: | -----------------------: | ---------------------: | --------------------: |
| MI300A regional |                     5.99% |                **6.60%** |                  4.55% |             **5.88%** |
| MI300A global   |                     4.81% |                **4.46%** |                  4.60% |             **4.31%** |
| GH200 regional  |                     2.06% |                **2.13%** |      0.88%, unresolved |             **1.77%** |
| GH200 global    |                     4.45% |                **4.56%** |                  4.23% |             **4.46%** |

All latest compute and wall gains pass the timing controls. In GH200 global
confirmation job 879318, compute fell from 31.730 to 30.282 ms and granule wall
from 32.525 to 31.075 ms. The wall saving was 1.450 ms against a 0.060 ms
identical-code noise threshold; its control-adjusted 95% interval was
[1.426, 1.473] ms. All 12 wall contrasts were positive, with no detected order
sensitivity. Code hashes, installed packages, initial inputs, node and timing
schedule matched the preceding run.

The preceding GH200 global run, 879091, remains an unresolved observation:
4.65% lower wall time, but one identical-code control block took 101 ms instead
of roughly 33 ms. The confirmation establishes a new controlled result; no
samples were removed from either run. GH200 regional's earlier sequential-run
11.59% wall increase also did not recur in the paired run; its cause remains unknown.

The 11–13% figures describe **individual programs inside the granule**:

| GPU / mesh      | Two solvers: earlier reduction | Two solvers: latest reduction | Theta-rho: earlier change   | Theta-rho: latest change      |
| --------------- | -----------------------------: | ----------------------------: | --------------------------- | ----------------------------- |
| MI300A regional |                         11.72% |                    **12.48%** | 13.37% reduction            | **13.83% reduction**          |
| MI300A global   |                         11.68% |                    **11.64%** | 0.29% increase; unresolved  | 0.53% increase; unresolved    |
| GH200 regional  |                          4.81% |                     **4.73%** | Saving, but order-sensitive | **2.63% reduction; resolved** |
| GH200 global    |                         11.19% |                    **11.39%** | 0.16% increase; resolved    | 0.002% reduction; unresolved  |

These are program contributions within the combined treatment, not independent
single-pass measurements. The solver percentage uses their summed times. On AMD
regional, the three target programs occupied about half the original compute
time: improving that half by roughly 12–14% gives 6.6% for the whole granule.

Earlier generated-code inspection found theta-rho **6 → 5 kernels on regional**
and **3 → 3 on global**, on both GPUs. Each solver specialization lost one kernel
and three temporary arrays, while explicit-wind preparation remained outside
the forward scan. The latest timing runs did not recount kernels. Earlier direct
compiler/frontend comparisons found no resolved performance difference; the
compiler gains replace the frontend gains and must not be added to them.

## How the measurements differ, and why

| Runs                                                                        | Measurement procedure                                                                                                                                                            | Purpose and limitation                                                                                                                                |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Earlier GLOBAL_REVIEW results                                               | Separate experiment plugin; synchronized wall calls timed with `perf_counter_ns`, plus GT4Py program timers; restored, alternating A/B blocks and A/A controls.                  | Controlled evidence for the compiler changes, using experimental infrastructure.                                                                      |
| Intermediate ordinary checks, AMD 646116 / GH200 878418                     | Normal pytest benchmark, separate off then on processes; measured-call means, without restored alternating blocks or A/A controls.                                               | Checked the normal model path, but small differences could be mixed with run order and changing state. This produced the GH200 regional disagreement. |
| Latest paired checks, AMD 647055 / GH200 879091; global confirmation 879318 | Existing dycore test and normal model options; `pytest-benchmark.pedantic` supplies the synchronized wall timer, with the same GT4Py program timers and controlled block design. | Keeps the earlier comparison controls while using the normal benchmark and production compiler configuration.                                         |

The earlier and latest paired designs use input seed 20260910, order seed
20260915, **12 balanced ABBA/BAAB quartets**, interleaved identical-code controls,
**five warmups and ten measured calls per block**. A is original; B enables both
passes. State is restored before each block. Compilation, restoration, warmup
and metrics bookkeeping are outside the timed interval. Block medians are
combined into quartet contrasts; uncertainty is calculated across quartets,
not by treating all calls as independent runs.

A result is resolved only if its 95% interval excludes zero, the size of the change exceeds
the identical-code noise threshold, and no order sensitivity is detected. This
rule was kept for the latest results. The new analysis also reproduced the
archived means and intervals from their raw samples. Harness and node differences
remain, so reproduction means comparable gains under a documented procedure,
not identical milliseconds or percentages on every node.

## Why regional benefits more on MI300A

The difference is specific to regional: **global solver gains are almost equal**
on the two GPUs. Fusion combines compatible calculations and avoids writing
some intermediate results to full arrays before reading them back. The cost
saved depends on memory access, launches and the resource use of the fused code.

Earlier [profiling](DYCORE_GRANULE_ANALYSIS.md) found better small-grid cache
reuse on GH200 and excess regional theta-rho traffic on AMD. It is plausible
that GH200 already handles some small intermediate working sets more efficiently,
so removing them saves less. Timing confirms a larger regional benefit on AMD;
it does not separate cache capacity, register use, scheduling and launch costs
into individual causes.

## Reproduce and next step

Use [the reproduction guide](REPRODUCE_DYCORE_OPTIMIZATIONS.md), keeping the usual
vendor environment and adding `--dycore-compare=combined` to the existing dycore
benchmark. `theta` and `solver` select either pass separately. Run each mesh once;
the comparison switches off/on internally. Read
`benchmarks[0].extra_info.comparison` in its JSON, **not the pooled pytest table**.

The paired jobs, including confirmation 879318, used ICON4Py model packages
`d51c1027d`, GT4Py `403f9d99`, DaCe `5115128a`, and the paired testing helper including its scalar-array restoration
fix. All **148 checked fields matched with maximum absolute difference zero**;
cached timestep/CFL values were restored, and source checks passed. The testing
helper in this change includes the GPU-tested scalar-array restoration fix.

The latest raw JSONs are in `amd_scripts/dycore_runs/`:
`mi300a_{regional,global}120_paired_647055_nid002424.json` and
`gh200_{regional,global}120_paired_879091_nid005260.json`, plus the confirmation
`gh200_global120_paired_879318_nid005260.json`; per-program timer files are at the
repo root. Earlier compact evidence is in
[COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json), with regional detail
in [REVIEW.md](REVIEW.md). Raw block medians, sums and intervals were checked
independently for the latest jobs.

Proceed with review of the two opt-in passes and the tested benchmark integration.
The GH200 global confirmation closes the remaining wall-time measurement
question. No further run is needed for this comparison; it does not settle the
separate question of how much of the saving comes from each hardware mechanism.
