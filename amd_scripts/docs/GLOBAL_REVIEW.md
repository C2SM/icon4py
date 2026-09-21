# Global/120 compiler regression validation — 2026-09-21

The combined compiler passes improve total solve_nonhydro granule performance
on both GPUs. Global correctness passes, and no compiler-versus-frontend
performance difference clears the conservative identical-arm control threshold.
This completes the planned global check for the opt-in PR; it is not universal
validation of all configurations.

AMD job 644950, nid002934; NVIDIA job 875596, nid005231. Both COMPLETE markers and
final status records are present. Mesh: 327,680 cells, 491,520 edges, 163,842
vertices, 120 levels, limited_area=False. Each comparison uses 12 balanced
quartets and interleaved identical-arm controls.

| Original to both compiler passes  | Original ms | Compiler ms | Reduction |
| --------------------------------- | ----------: | ----------: | --------: |
| MI300A summed program device time |   35.632809 |   33.919402 | **4.81%** |
| GH200 summed program device time  |   31.918701 |   30.499036 | **4.45%** |
| MI300A granule wall time          |   36.543464 |   34.862733 | **4.60%** |
| GH200 granule wall time           |   32.816230 |   31.427213 | **4.23%** |

All four total contrasts are positive in all 12 quartets, clear the conservative
control threshold, and show no detected order dependence. Device saving raw
95% intervals: AMD [1.667498, 1.759317] ms; NVIDIA [1.413344, 1.425986] ms.
Control-adjusted intervals remain positive: AMD [1.647030, 1.770042], NVIDIA
[1.410049, 1.422505] ms. These times exclude diffusion and the rest of a model
timestep.

## Global versus regional: combined compiler benefit

Both compiler passes together reduce whole-granule GPU time on both meshes and
both GPUs, with all 148 checked fields matching exactly. The regional grid has
44,528 cells; the global grid has 327,680. Both use 120 levels. Each percentage
compares the original and optimized code on the same GPU within one job.

| GPU    | Mesh     | Original GPU ms | Optimized GPU ms | GPU-time reduction | Original wall ms | Optimized wall ms |        Wall-time reduction |
| ------ | -------- | --------------: | ---------------: | -----------------: | ---------------: | ----------------: | -------------------------: |
| MI300A | Regional |        5.426298 |         5.101430 |          **5.99%** |         6.343732 |          6.055173 |                  **4.55%** |
| MI300A | Global   |       35.632809 |        33.919402 |          **4.81%** |        36.543464 |         34.862733 |                  **4.60%** |
| GH200  | Regional |        3.857321 |         3.777862 |          **2.06%** |         4.903749 |          4.860488 | 0.88% observed; unresolved |
| GH200  | Global   |       31.918701 |        30.499036 |          **4.45%** |        32.816230 |         31.427213 |                  **4.23%** |

GPU time means summed per-program device time. Wall time includes the host-side
cost of invoking the granule. All four GPU-time gains pass the identical-arm
controls, as do three wall-time gains; GH200 regional wall time does not clear
the noise threshold. These measurements cover solve_nonhydro, not a full model
timestep. Percentages are time reductions, calculated as `(original - optimized) / original`.

The solver improvement is shared across meshes. Theta-rho provides an additional
regional opportunity because its generated kernel structure differs:

| Program or structural change    | MI300A regional      | MI300A global              | GH200 regional                              | GH200 global             |
| ------------------------------- | -------------------- | -------------------------- | ------------------------------------------- | ------------------------ |
| Two solvers, GPU-time reduction | **11.72%**           | **11.68%**                 | **4.81%**                                   | **11.19%**               |
| Theta-rho, GPU-time change      | **13.37% reduction** | 0.29% increase; unresolved | Positive saving, but sensitive to run order | 0.16% increase; resolved |
| Theta-rho kernel count          | 6 → 5                | 3 → 3                      | 6 → 5                                       | 3 → 3                    |

These program timings are contributions within the combined treatment, not
separate measurements of each pass. The global solver gain outweighs the small
GH200 theta-rho slowdown. On both meshes, the compiler implementation recovers
the frontend implementation's benefit without a resolved performance difference;
these gains must not be added to earlier frontend gains.

This is a strong result for the tested configurations: the optimizations help
both vendors and both meshes without requiring the scientist to rewrite the
model equations. Each mesh used different nodes, so the difference between its
percentage gains is descriptive, not a controlled measurement of the mesh's
influence. It also does not establish cache capacity as the cause of the original
vendor gap. Regional provenance and timing controls are in [REVIEW.md](REVIEW.md);
global provenance and controls are recorded above and below.

## Which work improved

The two solver programs account for essentially all the global device saving:
AMD 14.186597→12.529389 ms (**11.68%**); NVIDIA 12.778721→11.349062 (**11.19%**).
Their generated-code records show one fewer kernel and three fewer global
temporary arrays in each specialization. Explicit-wind preparation stays outside
the sequential forward scan. Compiler/frontend kernel-count and temporary-shape
multisets match for both solvers.

Theta remains at three kernels, including two fused pressure/velocity kernels,
and four global temporary arrays on both GPUs. The regional 6→5 kernel change
is absent here. AMD theta's observed +0.29% time is unresolved. NVIDIA theta is
0.006793 ms (**0.16%**) slower in the combined arm, raw interval for saving
[-0.008113, -0.005473] ms, above its 0.002407 ms control threshold; control
adjustment retains the negative sign. Keep this small per-program regression in
the record. These combined-arm measurements do not establish that the theta pass
caused it, and the positive whole-granule result does not mean every program
improved.

The data now show solver fusion helps both meshes; the additional theta
structural benefit is regional. No cache-capacity attribution follows from this
comparison.

## Compiler versus frontend solver (theta fixed)

| GPU    | Frontend granule device ms | Compiler granule device ms | Saving and raw 95% interval, ms  |
| ------ | -------------------------: | -------------------------: | -------------------------------- |
| MI300A |                  33.990737 |                  33.996850 | -0.006114 [-0.042620, +0.030393] |
| GH200  |                  30.481849 |                  30.478152 | +0.003697 [-0.002127, +0.009522] |

Neither device contrast clears its control threshold. Some NVIDIA
control-adjusted or wall-time intervals are positive, but their contrasts also
remain below the conservative control thresholds; no extra compiler benefit is
claimed. This is comparable observed performance, not formal equivalence.

## Validation and reproducibility

All four comparisons pass **148 fields exactly**, covering 3,645,277,412 finite
values per comparison, maximum absolute error zero, plus scalar-state checks.
Inputs match between comparisons within each device. Source hashes before/after
are identical. All 24 recorded bundle source hashes match the archived sources.
The region is independently checked through reported grid identity, levels and
limited-area flag.

`python3 review_results.py` independently rebuilds block medians, program sums,
quartet contrasts and intervals for both meshes. All agree with the reports;
results are in `COMPILER_FUSION_RESULTS.json`. The source snapshot and raw reports are in each job
directory. This review used only rsync for cluster access. No cluster jobs were executed
and no compiler implementation was changed during the review.

## Published evidence and archived captures

The compact reconstructed results are included in [COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json). The `review_results.py` checker, raw job directories, and frozen experiment sources mentioned above remain in the original `amd_scripts/compiler_stage_fusion_runs/` archive; they are not bundled into this small PR. [Current-branch setup and benchmark checks](REPRODUCE_DYCORE_OPTIMIZATIONS.md) use the published compiler dependency, without applying copied compiler patches.
