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

## How the 48% is measured

For each repeatedly called program, we take its median GPU execution time after
five warmup calls, over 50 ordinary timing calls. We then add those medians:

```text
MI300A: 5.735138 ms     GH200: 3.8773835 ms
Extra device time: 5.735138 − 3.8773835 = 1.8577545 ms
Relative extra time: 1.8577545 / 3.8773835 = 47.91%
```

This is a sum of typical program times, not one stopwatch interval around the
entire GPU call. Counter-profiler durations are not used for this headline.
The reference pair is AMD job 631263 on nid002728 (September 10) and GH200 job
855892 on nid005083 (September 9), with both grids measured in each job.
The main comparison is therefore not an old global AMD run divided by a fresh
regional NVIDIA run.

Whole-call host-wall timing includes the work needed to launch and finish the
GPU calculations. Regional is **6.4133 versus 4.7893 ms**, or **1.339×: 33.9% more
time**. That is why we specify the measurement when quoting 48%. Neither number
is an end-to-end forecast for the weather model. These separately aggregated
summaries also cannot be subtracted to isolate an exact Python-overhead cost.

The grid dependence repeats across allocations: the September 9 regional pair
was 1.448×, this pair is 1.479×, and the later unchanged-code control gave 1.485×.
Treat 48% as the representative result of this comparison, not a universal
constant for every node or run. Regional still takes much less absolute time
than global on both chips; **GH200 gains more from moving to regional**.

## Which calculations account for the extra time?

Subtracting GH200's program times from AMD's gives a useful accounting of the
1.858 ms. It tells us where to investigate, independently of cache hypotheses.

| Regional calculation | MI300A ms | GH200 ms | Extra ms | Share of total gap |
|---|---:|---:|---:|---:|
| Theta-rho / pressure-gradient / wind update | 1.041091 | 0.475281 | **0.565810** | **30.5%** |
| Predictor and corrector vertical solvers together | 1.897274 | 1.507057 | **0.390217** | **21.0%** |
| Corrector vertical momentum advection | 0.497702 | 0.265080 | 0.232623 | 12.5% |
| Horizontal velocity quantities and fluxes | 0.569663 | 0.394180 | 0.175484 | 9.4% |
| Horizontal momentum advection | 0.377937 | 0.213962 | 0.163976 | 8.8% |
| All remaining programs, including the tiny halo update | 1.351472 | 1.021825 | 0.329647 | 17.7% |

Theta-rho is **2.190×** GH200 on regional, versus **1.159×** on global. It is the
largest individual contributor, but cannot explain the other 69.5% of the gap.
The vertical solvers matter for a different reason: they consume about a third
of AMD's original regional device time and contribute another fifth of the gap.
Those two observations explain our optimisation priorities. A large slowdown
ratio on a tiny program is less important than milliseconds saved in a large one.

## Why the two meshes behave differently

Global has **327,680 cells**; regional has **44,528**. Regional is about one
seventh the size, but also has boundaries, different neighbour connections and
different ranges over which operations run. Size and structure change together.
We have not isolated their individual contributions by varying just one of them.

The generated theta-rho code illustrates the structural difference. Global has
**three kernels**, with interpolation joined to the pressure/wind updates.
Regional has **six kernels**: some interpolation, boundary and update work runs
separately. A kernel is a piece of work launched on the GPU. Separate kernels
can require intermediate fields to be written and read again.

Both vendors run the six-kernel regional version. Therefore “three extra
kernels” alone does not explain why AMD loses more: the GPUs must respond
differently to that work, its data access or the generated implementation.
Nor can we just remove boundary kernels; they perform required calculations.

## Worse locality and better cache reuse can both be true

Spatial locality asks whether values needed together are near each other in
memory. Cache reuse asks whether a value is still available in fast memory when
it is needed again. A smaller problem can improve the second even if its
neighbour accesses are more scattered.

Our edge-to-cell ordering analysis found more scatter on regional, including
its interior. Against the tested edge space-filling-curve sort, native regional
interior ordering touches **1.46×** as many distinct 64-byte lines in the proxy;
whole-grid global native ordering scores **0.79×** its corresponding sort.
These are geometric access estimates, not measured runtime improvements from
renumbering the mesh. They argue against blaming the halo alone.

The fixed-call cache collection supports a larger reuse improvement on GH200:

| Counter statistic over generated stencil kernels | Global | Regional | Change |
|---|---:|---:|---:|
| MI300A L2 hits / all requests | 29.26% | 38.08% | **+8.82 percentage points** |
| GH200 L2 hits / read+write sectors | 29.37% | 54.98% | **+25.61 points** |
| GH200 L2 hits / read sectors only | 8.84% | 42.88% | **+34.04 points** |

So yes: **AMD's mixed-request hit statistic improves much less between grids**.
That is relevant evidence, not something to dismiss. But requests and fixed-size
sectors are different counting units, and the read/write mix matters. An AMD
hit can also include a request merged with an outstanding fill. We cannot
interpret the percentage difference as a measured quantity of extra HBM traffic,
cache residency or milliseconds lost. Similar global percentages do not calibrate
the counters across vendors. An AMD read-specific ratio derived from outgoing
fabric requests remains a conditional proxy, not an equivalent measured read-hit rate.

Closer to the execution units, AMD's L1 read-forwarding proxy rises from
**14.67% to 17.90%**, and tag accesses per state-read request rise from **18.11
to 20.86**. These support investigating scattered accesses and retention, but
do not themselves quantify time lost. GH200's L1 load-hit statistic changes
only modestly, **38.59% to 37.90%**, in this same fixed-call collection.

Flushing GH200 caches before each profiled kernel still leaves regional L2 read
hits at **42.27%**, versus **42.88%** when caches are preserved; global remains
near 8.8%. Thus the regional advantage largely survives without data inherited
from previous kernels. Reuse within a kernel is a stronger candidate than
cross-kernel cache carry-over. This does not prove that the whole regional
working set fits in L2. These values exclude library kernels; older full-capture
figures around 42.0%/42.7% describe the same trend with a slightly wider scope.

## Traffic helps identify targets, not assign the entire timing gap

On AMD, compare each program's regional reads with its global reads scaled by
the number of relevant cells or edges. Theta-rho demand near L1 scales roughly
with edge count, but reads forwarded towards L2 are about **1.83×** that simple
reference, and reads beyond L2 are **1.61×**. The extra traffic appears further
down the memory path; it is not explained by doing 1.61× as many initial requests.

| Program | Regional reads beyond L2 | Reference scaled from global | Positive excess |
|---|---:|---:|---:|
| Theta-rho | 2.178 GB/call | 1.350 GB/call | **0.829 GB/call** |
| Corrector vertical momentum | 1.152 GB/call | 0.692 GB/call | **0.459 GB/call** |

Together these account for **79.2% of the 1.626 GB/call positive excess** across
programs common to both meshes. That is a traffic-ranking statistic, **not 79.2%
of the runtime gap**. Programs below the reference offset some of it; net extra
reads are about 0.740 GB/call. Scaling by cell/edge count does not equalise active
domains, boundary work or instructions, so not all excess is necessarily avoidable.

These AMD counters measure traffic leaving L2 towards the fabric. Another cache,
MALL, can serve some of it before HBM, the main GPU memory. GH200 HBM counters
measure at a different boundary. This is why the bandwidth accounting could not
reliably turn the byte difference into a share of the 1.858 ms. The separate
2,951 GB/s AMD and 3,743 GB/s GH200 streaming tests were useful measurements,
but not validated universal ceilings for these kernels.

## Latency and occupancy: evidence of waiting, not a complete cause

In the later control run, AMD's time-weighted memory-instruction completion
metric rises from about **2,016 to 2,991 clock ticks** on regional. Its resident
waves per compute unit fall from **19.7 to 16.1**. A wave is a group of threads;
keeping more waves ready can help execute useful work while others await data.
These observations make memory waiting and the ability to hide it plausible
contributors.

But the main target, theta-rho, has **more** resident waves on regional
(**12.7 → 16.3**), alongside a longer completion metric (**1,373 → 3,302**).
More residency is not necessarily more progress: threads can stay resident
while waiting. Conversely, a rise in latency does not identify which cache or
memory component caused it.

AMD's L2-busy metric falls **96.9% → 79.5%** across the granule, and
**98.6% → 68.6%** for theta-rho. It measures cycles with requests pending at L2;
it does **not** establish that 20–31% of total runtime is unused and recoverable.
These diagnostic averages come from separate profiler passes and weight kernels
by their profiled duration. They cannot be read as an elapsed-time decomposition.

## What the intervention actually established

We went beyond correlations by changing theta-rho's structure and measuring
ordinary execution time with unchanged inputs. The repeated Python-domain-fusion
experiment (AMD job 634088 / GH200 job 861414) gave:

- AMD granule: **5.472667 → 5.334810 ms**, a **2.52% reduction**.
- GH200 granule: **3.832493 → 3.846001 ms**, a **0.35% increase**.
- The vendor gap shrank by **0.151366 ms**, **9.23% of that run pair's gap**;
  the 95% interval for the differential saving was **[0.118626, 0.184106] ms**.

This establishes that a code-structure change can remove a measurable part of
the regional vendor disadvantage. AMD theta-rho fabric reads fell **11.20%**
under matched native/fused profiling, and its completion-time metric fell too.
Occupancy approximately halved while execution got faster, so “increase
occupancy” is not an adequate explanation or prescription on its own.

The intervention changed several things together. It does not establish that
all saved time came from fewer bytes or cache capacity. In fact, that Python
rewrite produced seven total kernels rather than six; “fusion helps” did not
mean “fewer total launches” in that experiment. GH200 read/write traffic also
fell slightly while its execution time increased.

The compiler transformation below is the cleaner successor: it handles the
original program and produces five regional kernels without the Python rewrite's
duplicated buffers. The earlier measured **9.23% gap closure belongs to the
Python intervention and its own baseline**, not automatically to the new compiler
or combined solver result. That historical implementation is not added to this branch.

**Our supported explanation is now more specific:** regional changes both
access patterns and the way calculations are separated into kernels; the two
GPUs respond differently to those changes. Traffic and reuse measurements locate
plausible losses, and a controlled structural change has reduced part of the gap.
We still cannot allocate the remaining milliseconds uniquely among locality,
cache capacity, latency hiding, launch overhead and other code-generation effects.

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

This review branch is based directly on **C2SM's `mi300_opt`**, commit
`397d774a17135702b411d97edd4fb42cd0e21566`, the model base used for the measurements.
It retains that branch's vertical-level interface, workspace and vendor tuning;
it is not a port to upstream `main`. The solver scan, field operator and public
program match the measured prototype structurally, ignoring docstrings, while
the existing standalone scan API is retained for compatibility.

The restored implementation passes six isolated embedded/compiled CPU
comparisons against the independent NumPy reference at 2, 40 and 120 levels
in double precision. This complements the recorded regional GPU validation;
it does not establish global-grid, mixed-precision or combined performance.
The PR base should be **C2SM/icon4py:mi300_opt**, so reviewers see only the
optimisation, tests, explanation and reproduction entry points.

The [GT4Py patch](../../patches/gt4py-shared-output-fusion.patch) contains the compiler
transformation and regression tests (22 focused tests and pre-commit checks pass).
GT4Py is a separate repository, so the change is carried here as an applyable
patch rather than copying the compiler into Icon4Py. Its base is GT4Py
`a461b874` (upstream main); the modified transformation source is
unchanged from the measured prototype. It does not automatically enable theta
fusion in Icon4Py. The same compiler change is also committed separately as
`857e718d` on GT4Py branch `dycore-shared-output-fusion`, ready to push to
`dganellari/gt4py` for a normal compiler PR.

This branch contains the solver change, depth-parameterised
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

## Provenance of the starting comparison and diagnostics

The fixed-call timing and cache numbers above come from AMD job 631263
(`20260910T103804Z_mi300a_nid002728_631263`) and GH200 job 855892
(`20260909T192629Z_nid005083_855892`). Program medians and the gap breakdown were
recomputed from `comparison.json` and `GRID_EVIDENCE.json`; cache rates use
`COUNTER_SUMMARY.json` and its raw-count audit. Device totals include the tiny
halo program; the counter tables cover generated stencil kernels only (52 global,
60 regional), three captured calls after five warmups. Collection succeeded;
final parsing failures were recovered locally, and failed batch statuses were retained.

Latency/occupancy diagnostics come from the September 11 A/A pilot, AMD 632411
and GH200 858442. Its regional timing control passed; the global control had
bias, and a final source-inventory check failed after two analysis files were
added. A later audit found the original files unchanged, but cannot exclude
transient changes during the run. These limitations are why its counters are
used as diagnostics rather than a causal time breakdown.

The historical fusion intervention is independently reviewed in
`causal_runs/fusion_review_2026-09-14/REVIEW.md` on the original local experiment
archive. Full profiling captures remain separate from this small review branch;
the numerical tables here retain the definitions, run identities and limitations
needed to interpret the results. The later compiler/solver timing evidence and
replay instructions are included through the pinned experiment bundle.
