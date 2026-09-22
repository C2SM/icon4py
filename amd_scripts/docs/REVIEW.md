# Restricted compiler fusion: regional GPU validation

The compiler passes recover the measured frontend benefit on this regional/120 test. There is no resolved performance difference between the compiler solver and the frontend solver. This replaces the frontend implementation of the optimization; it is not another 6% to add to it. These regional jobs do not test global performance; the completed follow-up is documented in [GLOBAL_REVIEW.md](GLOBAL_REVIEW.md).

MI300A: job 641726, nid002706. GH200: job 873329, nid005017. Both COMPLETE markers and final status records are present. Each comparison uses 12 balanced ABBA/BAAB quartets with interleaved identical-arm controls. Times below cover solve_nonhydro only, not diffusion or a full model timestep. Device time is the sum of per-program device-time medians; wall time measures the granule invocation.

| Original to both compiler passes | Original device ms | Compiler device ms | Device-time reduction | Original wall ms | Compiler wall ms |
| -------------------------------- | -----------------: | -----------------: | --------------------: | ---------------: | ---------------: |
| MI300A                           |           5.426298 |           5.101430 |             **5.99%** |         6.343732 |         6.055173 |
| GH200                            |           3.857321 |           3.777862 |             **2.06%** |         4.903749 |         4.860488 |

MI300A wall time falls **4.55%**, also resolved above the conservative control threshold. GH200 wall time falls an observed 0.88%, but its 0.04326 ms saving is below the 0.06347 ms control threshold; do not claim a resolved wall-time benefit there. Both vendors' total device savings are positive in all 12 quartets and show no detected order dependence. Raw 95% saving intervals: AMD [0.313325, 0.336411] ms; GH200 [0.077901, 0.081015] ms. Subtracting matched control contrasts also leaves positive intervals: AMD [0.303136, 0.330450], GH200 [0.076059, 0.081984] ms.

Within the combined treatment, AMD's two solvers improve 11.72% and theta-rho 13.37%; GH200's solvers improve 4.81%. These are per-program contributions within the combined arm, not independent isolated-intervention measurements. GH200 theta-rho has a positive contrast but its magnitude is order-sensitive.

The direct comparison keeps compiler theta fusion in both arms:

| Frontend solver to compiler solver | Frontend granule device ms | Compiler granule device ms | Saving and raw 95% interval, ms  |
| ---------------------------------- | -------------------------: | -------------------------: | -------------------------------- |
| MI300A                             |                   5.114034 |                   5.105543 | +0.008492 [-0.008766, +0.025749] |
| GH200                              |                   3.760003 |                   3.759129 | +0.000874 [-0.000901, +0.002649] |

Neither contrast clears the identical-arm noise threshold; control-adjusted intervals also include zero. Some tiny contrasts reverse sign with order. The supported conclusion is comparable observed performance with no detected advantage or regression, not formally established equivalence or an extra speedup.

All four comparisons validate **148 fields exactly**, maximum absolute error zero, plus scalar-state checks. Regional dimensions are 44,528 cells, 67,096 edges, 22,569 vertices and 120 levels. Initial inputs match between comparisons on each device. Source hashes are unchanged throughout each job, and all 24 recorded bundle source hashes match the archived candidate. Array fingerprints differ across vendors, so these are controlled within-device comparisons, not bit-identical cross-vendor input experiments.

I independently reconstructed block totals from raw sample per-program medians, then the quartet contrasts, arm averages and 95% intervals. The summaries agree. The reproducible check is `python3 review_results.py`; detailed results are in `COMPILER_FUSION_RESULTS.json` and the job subdirectories.

Generated-code audit records show that the solver keeps explicit-wind preparation outside the sequential forward scan. For all predictor/corrector specializations, compiler and frontend versions have matching kernel counts and temporary-array shape multisets. The narrower pass therefore removes the excess fusion seen in the earlier slower compiler trial and now reproduces frontend performance. Comparing those different-node trials does not isolate excess fusion as the sole cause of the earlier regression.

The regional performance gate for the compiler replacement is passed. The implementation is prepared on the GT4Py `dycore-fusion-passes` branch, with opt-in selection and safety guards. The planned global follow-up is now complete: see [GLOBAL_REVIEW.md](GLOBAL_REVIEW.md). No new jobs were submitted and no model/compiler sources were changed during this review.

## Published evidence and archived captures

The compact reconstructed results are included in [COMPILER_FUSION_RESULTS.json](COMPILER_FUSION_RESULTS.json). The `review_results.py` checker, raw job directories, and frozen experiment sources mentioned above remain in the original `amd_scripts/compiler_stage_fusion_runs/` archive; they are not bundled into this small PR. [Current-branch setup and benchmark checks](REPRODUCE_DYCORE_OPTIMIZATIONS.md) use the published compiler dependency, without applying copied compiler patches.
