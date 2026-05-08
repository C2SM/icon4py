# MI300A Profiling Results — vertically_implicit_solver_at_predictor_step

Date: 2026-03-30 (initial), 2026-04-19 (latest update)

> **This is the executive summary.** Detailed evidence lives in:
>
> - [docs/HARDWARE_REFERENCE.md](docs/HARDWARE_REFERENCE.md) — vendor specs + on-node sysinfo for MI300A and GH200
> - [docs/DEEP_ANALYSIS.md](docs/DEEP_ANALYSIS.md) — per-kernel deep dive on map_100_fieldop_1, cross-platform A/B tables, demand-byte derivation
> - [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md) — every optimization tried (block size, fusion, occupancy, loop blocking, LDS, CSE) with full A/B sweeps

## Summary

Profiled the `vertically_implicit_solver_at_predictor_step` GT4Py program on
MI300A (Beverin, gfx942, ROCm 7.1.0) and GH200 (Santis, sm_90).

### Headline result

A one-line config change — `gpu_block_size_2d=(256,1,1)` for ROCm in `model_options.py`
— gives **20% solver speedup** on MI300A and closes most of the MI300A vs GH200 gap.

| Platform | Config                            | GT4Py Timer median (1000 runs) |
| -------- | --------------------------------- | ------------------------------ |
| MI300A   | DaCe default `(32,8,1)`           | 0.753 ms                       |
| MI300A   | **`gpu_block_size_2d=(256,1,1)`** | **0.604 ms (-19.8%)**          |
| GH200    | DaCe default `(32,8,1)`           | 0.546 ms                       |
| GH200    | **`gpu_block_size_2d=(256,1,1)`** | **0.527 ms (-3.5%)**           |

GH200 numbers above are from the 2026-04-20 full sweep of 20 block sizes
(see [docs/ATTEMPTED_OPTIMIZATIONS.md#block-size-sweep-on-the-actual-solver-gh200-gt4py-timer-1000-runs](docs/ATTEMPTED_OPTIMIZATIONS.md#block-size-sweep-on-the-actual-solver-gh200-gt4py-timer-1000-runs)).
Earlier ~0.559 ms number was from an uncached default run; 0.546 ms here is the
clean A/B measurement.

Gap MI300A → GH200: **1.38x → 1.15x** (at each platform's best — MI300A 604 μs vs GH200 527 μs).
Previously reported 1.08x was MI300A-best vs GH200-default; with both at (256,1,1), the hardware-floor gap is ~15%.
Block-size tuning helps MI300A much more than GH200 — see
[Cross-platform table](docs/DEEP_ANALYSIS.md#cross-platform-memory-hierarchy-mi300a-vs-gh200--fully-verified-ab)
for per-kernel detail.

### Cross-platform story: AMD MI300A vs NVIDIA GH200 (in plain language)

**The big picture (all numbers verified):**

1. **MI300A is ~30% slower than GH200 per kernel today** (MI300A 166 μs vs
   GH200 116 μs on map_100_fieldop_1, each platform's best block size).
2. **The gap is mostly hardware**: GH200 has a higher HBM peak (4.0 vs 3.47 TB/s)
   AND saturates more of it (89% on GH200 vs 70% on MI300A on this kernel).
3. **Block size matters more on MI300A than on GH200**: switching (32,8)→(256,1,1)
   gives MI300A −24% on this kernel and ~−20% on the full solver. Same change on
   GH200 only gives −5% per-kernel / **−3.5% full solver** (measured 2026-04-20
   across a 20-variant sweep), because GH200 was already nearly bandwidth-saturated.
   Notable: **(256,1,1) is the sweet spot on BOTH platforms** — gating it inside the
   ROCM block only leaves ~3.5% on the table for GH200 users.
4. **Both architectures are bandwidth-bound, not compute-bound** on this kernel:
   IPC ~0.24 on MI300A, ~0.28 on GH200 (small fractions of each platform's peak);
   VALU usage 1-2% on both. The kernel waits on memory, not on math.
   *(IPC = Instructions Per Cycle. Low IPC means the scheduler issues an
   instruction only every 4-5 cycles → idle ~80% of the time. VALU = % cycles
   the vector ALUs compute. 1-2% means math units are idle 98% of the time
   waiting on memory. Implication: optimize memory traffic / cache hits /
   coalescing — not FLOPs.)*
5. **Both architectures' caches absorb a similar fraction (37-43%)** of demand bytes.
   At MI300A's optimal `(256,1,1)`, vL1D hit rate is **72%**; at GH200's optimal
   `(256,1,1)`, L1 hit rate is **only 13%**. So MI300A's per-CU L1 catches ~5×
   more cache hits than GH200's per-SM L1. GH200 wins overall because its HBM
   is faster anyway and absorption shifts to L2 / fabric instead.
   *(CU = Compute Unit, AMD's equivalent of NVIDIA's SM. vL1D is AMD's name for
   the per-CU L1 data cache — same role as NVIDIA's L1, just different name.)*

### Can MI300A catch GH200?

**The hardware-bound part of the gap is real and won't disappear.** GH200 is
already at 89% of its 4.0 TB/s HBM peak; MI300A is at 70% of its 3.47 TB/s peak.
Even if MI300A reached 100% of its HBM peak, the HBM-peak ratio alone (3.47 vs
4.0 = 0.87) would leave a residual gap.

**Where MI300A can still improve:**

- **HBM utilization at 70% vs GH200's 89%** — room to push more data per second.
- **Reduce HBM traffic** by avoiding kernel-to-kernel HBM round-trips on
  intermediate arrays (`gtir_tmp_83/96/97` etc.). Both platforms are HBM-bound,
  so cutting demand bytes pays off on both — though the *visible* speedup
  would be larger on MI300A (lower HBM utilization to start with) than on
  GH200 (already near saturation). Blocked by gt4py's `concat_where` pipeline
  (see Optimization Opportunities #2 below).

**Hardware floor (HBM-peak ratio):** MI300A 3.47 TB/s vs GH200 4.0 TB/s → **0.87**.
Even at perfect saturation, MI300A would still be ~13% behind from hardware
alone on bandwidth-bound kernels. Beating GH200 across the full model would
require per-kernel tuning (block size, registers, occupancy) since each kernel
has its own bottleneck profile.

**Note on C2E coalescing (commonly suspected, investigated 2026-04-20):** vL1D
coalescing on the heavy kernel is only 27% of peak. This sounds bad, but vL1D
itself is only 27.79% utilized (huge headroom in cache bandwidth) and the
kernel is HBM-bound, not vL1D-bound. Fixing C2E coalescing is predicted to
give \<2% wall-clock improvement. Details in
[Deep Analysis: C2E scatter](docs/DEEP_ANALYSIS.md#c2e-scatter-analysis).

### Why (256,1,1) helps

The default `(32,8)` thread block layout works fine for NVIDIA's 32-wide warps but
mismatches AMD's 64-wide wavefronts.

**Block layout reasoning:**

- All 256 threads in a block process consecutive cells → wavefront-aligned access,
  perfect coalescing for `array[cell, K]`.
- The default `(32,8,1)` spreads 8 threads across K levels, which wastes coalescing
  potential since K is the non-unit-stride dimension.
- MI300A wavefronts are 64 wide, so 256 = 4 wavefronts per block — good occupancy.

**Verified performance impact:**

- Cell-consecutive threads on the same CU dramatically improve cache reuse:
  **L2 hit rate jumps 15-20% → 32-50%** on the heavy 2D kernels.
- vL1D hit rate reaches **72%** on heavy kernels (per-CU TCP catches reuse).
- Per-kernel duration drops 18-24%.
- Scan kernels (1D maps) are unchanged.

**Note:** even with `(256,1,1)`, **vL1D coalescing is still only 27% of peak**
on the heavy kernel — the C2E gather scatters threads across edge memory.
But this turns out NOT to be a wall-clock bottleneck (vL1D has plenty of
bandwidth headroom; the kernel is HBM-bound, not vL1D-bound). See
[Deep Analysis: C2E scatter](docs/DEEP_ANALYSIS.md#c2e-scatter-analysis) for
why fixing this would give \<2%.

## Optimization Opportunities

### Confirmed helpful

1. **`gpu_block_size_2d=(256,1,1)` for ROCm** — −20% solver speedup on MI300A,
   −5% per-kernel on GH200 (see [block size tuning detail](docs/ATTEMPTED_OPTIMIZATIONS.md#gpu-block-size-tuning))

### Worth investigating

2. **Fuse `gtir_tmp_83/96/97` and similar intermediates** — these arrays are
   written to HBM by one kernel and read back by the next. Both platforms are
   bandwidth-bound (MI300A 70% of peak, GH200 89%), so removing this round-trip
   directly reduces HBM traffic on both. **Largest remaining opportunity on
   map_100_fieldop_1.** Likely larger visible speedup on MI300A (more HBM
   headroom) than on GH200 (already near saturation), but helps both. Blocked
   by `concat_where` K-domain splits in DaCe (GT4Py pipeline concern, not a
   tuning knob). See
   [docs/ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating](docs/ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating).
3. **Grid size divisible by 6** — preliminary single-data-point shows ~4%
   additional improvement (0.617→0.594 ms on a different gt4py branch). Possibly
   related to even work distribution across MI300A's 6 XCDs. Needs A/B verification.
4. **XCD-aware block placement** — orthogonal to block size; on MI300A the
   block-to-XCD assignment affects load balance and per-XCD L2 reuse. Could
   stack on top of (256,1,1).

### Not helpful (verified, with measurements)

| Optimization                                                    | Result                                                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loop blocking (K-blocking)                                      | 2.8% on its own, redundant once `(256,1,1)` is applied                                                                                                                                                                                                                                 |
| `fuse_tasklets`                                                 | Neutral on current gt4py (was 7% on older 1.1.4)                                                                                                                                                                                                                                       |
| Kernel fusion (for launch overhead)                             | Inter-kernel gap is only 0.5-1.1% — not worth pursuing                                                                                                                                                                                                                                 |
| Forcing higher occupancy                                        | 2.5x slowdown (register spilling); 10/12 kernels already at max                                                                                                                                                                                                                        |
| Register limiting via compiler flags                            | Not exposed in ROCm 7.1.0                                                                                                                                                                                                                                                              |
| CSE (common subexpression elimination)                          | Compiler handles it at -O3                                                                                                                                                                                                                                                             |
| Block size `(64,6,1)`                                           | 7% worse than `(256,1,1)`                                                                                                                                                                                                                                                              |
| LDS / shared-memory staging of the C2E gather                   | Neutral in synthetic test (no inter-thread reuse to exploit). `__shfl`-based deduplication is untested but targets the same vL1D-only metric — same caveat as C2E reordering applies (kernel is HBM-bound, predicted \<2% wall-clock impact).                                          |
| Many-array BW limitation                                        | Synthetic benchmark proves MI300A handles 20+ arrays at 93% peak                                                                                                                                                                                                                       |
| **C2E edge reordering** (downgraded from "worth investigating") | Predicted \<2% wall-clock impact. The "27% vL1D coalescing" number wastes vL1D bandwidth, but vL1D is only at 27.79% utilization (huge headroom). The kernel is HBM-bound, not vL1D-bound. Reordering improves a metric that isn't the bottleneck. See "How to verify yourself" below. |

Full A/B sweeps and methodology in [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md).

### How to verify yourself

These are the commands behind the key claims above. Run on the relevant cluster
after the standard env setup (`source amd_scripts/setup_env.sh; source .venv_rocm/bin/activate`
on MI300A; `source .venv_cuda/bin/activate` on GH200).

**Block-size A/B (MI300A, GT4Py Timer):**

```bash
# Edit model_options.py to set the block size you want (or comment line for DaCe default)
# Use a fresh GT4PY_BUILD_CACHE_DIR per variant — gt4py caches by SDFG hash, not by block size
export GT4PY_BUILD_CACHE_DIR=amd_blocksize_<variant>_solver_regional
export GT4PY_COLLECT_METRICS_LEVEL=10
export GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE="1"
export ICON4PY_STENCIL_TEST_WARMUP_ROUNDS=3
export ICON4PY_STENCIL_TEST_ITERATIONS=10
export ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS=100

srun --partition=mi300 --gres=gpu:1 --ntasks=1 --time=00:15:00 \
    .venv_rocm/bin/python -m pytest -sv -m continuous_benchmarking -p no:tach \
    --backend=dace_gpu --grid=icon_benchmark_regional \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"

# Verify the actually-compiled block size:
LAST=$(ls -td $GT4PY_BUILD_CACHE_DIR/.gt4py_cache/*/ | head -1)
grep -oE "dim3\([^)]*\), dim3\([^)]*\)" $LAST/src/cuda/hip/*.cpp | sort -u | head -3
```

**Per-kernel coalescing / cache metrics (MI300A, rocprof-compute analyze):**

```bash
# Find the dispatch IDs of the kernel of interest
P=workloads/rcu_amd_256x1_solver/MI300A_A1
awk -F',' 'NR>1 && /map_100_fieldop_1/ {print $1}' $P/pmc_perf.csv | sort -u

# Run the analyze (replace 11 with one of the dispatch IDs)
rocprof-compute analyze -p $P --dispatch 11 > /tmp/m100.txt 2>&1

# Coalescing, hit rates, L1-L2 traffic:
sed -n '/^16\. Vector L1/,/^17\./p' /tmp/m100.txt   # vL1D section: coalescing, hit rate, L1-L2 BW
sed -n '/^2\.1 System/,/^2\.2/p' /tmp/m100.txt      # SOL: HBM BW, IPC, VALU
sed -n '/^17\. L2 Cache/,/^18\./p' /tmp/m100.txt    # L2 section: hit rate, fabric BW
```

**Per-kernel L1↔L2, L2-Fabric, HBM BW + L2 hit rate (MI300A, lighter-weight):**

```bash
python3 amd_scripts/extract_pmc.py workloads/<your-workload>/MI300A_A1/pmc_perf.csv
```

Reports three cache-plane bandwidths per kernel. HBM column matches
`rocprof-compute analyze` §4.1.9 within ~0.4% (validated 2026-05-03).
See module docstring in `extract_pmc.py` for formulas.

**Inter-kernel gap (MI300A, rocprofv3 kernel-trace):**

```bash
srun --partition=mi300 --gres=gpu:1 --ntasks=1 --time=00:15:00 \
    rocprofv3 --kernel-trace --output-format csv -o rocprofv3_<variant> -- \
    .venv_rocm/bin/python -m pytest -sv -m continuous_benchmarking -p no:tach \
    --backend=dace_gpu --grid=icon_benchmark_regional \
    model/atmosphere/dycore/tests/dycore/stencil_tests/test_vertically_implicit_dycore_solver_at_predictor_step.py \
    -k "test_TestVerticallyImplicitSolverAtPredictorStep[compile_time_domain-at_first_substep[False]__is_iau_active[False]__divdamp_type[32]]"
# Then post-process: wall_clock(first→last kernel) − Σ kernel_durations per iteration
```

**Cross-platform DRAM bytes (GH200, ncu):**

```bash
# Re-query existing .ncu-rep without re-running the kernel:
ncu --import gh200_solver.ncu-rep --csv \
    --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
    -k regex:'map_100_fieldop_1' 2>&1 | grep -E "dram__bytes|gpu__time"

# To re-run on a different cache (sbatch on santis):
sbatch <wrapper>  # see amd_scripts/profile_solver_gh200_ncu.sh for template
```

**C2E scatter analysis (grid-only first-order estimate, NOT a measurement):**

```bash
python3 amd_scripts/analyze_c2e_reorder.py testdata/grids/mch_opr_r19b08/domain1_DOM01.nc \
    --wave-size 64 --cells-per-block 256
```

⚠️ The script's model is approximate. The cache-line-utilization numbers
(>100% in some configs) prove the model is misformulated — useful only as a
rough estimate of edge-index spread, not as a coalescing measurement. The
authoritative numbers come from `rocprof-compute analyze` (above).

## Per-Kernel Timing (rocprofv3 kernel-trace, median μs, MI300A)

Both columns measured with rocprofv3 kernel-trace, fresh runs on the current gt4py.

| Kernel                      | Baseline (μs) | (256,1,1) (μs) | Improvement | Stencil                                      |
| --------------------------- | ------------- | -------------- | ----------- | -------------------------------------------- |
| map_100_1                   | 212           | **166**        | -22%        | thermo results + dwdz (split 1)              |
| map_111_1                   | 186           | **146**        | -22%        | explicit rho/exner + divergence (split 1)    |
| map_60                      | 138           | **112**        | -19%        | solver coefficients + w explicit             |
| map_0                       | 60            | 50             | -16%        | contravariant correction (C2E gather)        |
| map_85                      | 58            | 57             | neutral     | tridiag forward sweep (scan, 1D)             |
| map_31                      | 42            | 33             | -21%        | w explicit term                              |
| map_90                      | 27            | 27             | neutral     | tridiag back-sub (scan, 1D)                  |
| map_91                      | 9             | 7              | -22%        | Rayleigh damping                             |
| map_100_0                   | 6             | 6              | neutral     | thermo results split 0 (1D)                  |
| map_111_0                   | 5             | 5              | neutral     | rho/exner split 0 (1D)                       |
| map_13                      | 6             | 6              | neutral     | contravariant correction lower boundary (1D) |
| map_35                      | 4             | 4              | neutral     | zeroing temporaries (1D)                     |
| **Total kernels**           | **753**       | **618**        | **-18%**    | sum of per-kernel medians                    |
| **Wall clock (first→last)** | 774           | 660            | -15%        | inter-kernel gap: 0.5-1.1%                   |
| **GT4Py Timer median**      | 760           | 604            | -21%        | full SDFG call (1000 runs)                   |

The 2D heavy kernels (map_100_1, map_111_1, map_60, map_31) all gained 16-22% from
the (256,1,1) block size. 1D kernels and scans are unchanged as expected.

### Per-Kernel Cross-Platform (MI300A vs GH200 at each platform's best block size)

MI300A at `(256,1,1)`, GH200 cache compiled with (32,8) for 1D/scan kernels and
(256,1,1) for 2D kernels (DaCe's automatic per-kernel choice on GH200).

| Kernel                        | MI300A (μs) | GH200 (μs) | Winner            | Note                  |
| ----------------------------- | ----------- | ---------- | ----------------- | --------------------- |
| map_100_1                     | 166         | 117        | GH200 (-30%)      | hottest 2D kernel     |
| map_111_1                     | 146         | 107        | GH200 (-27%)      | 2D                    |
| map_60                        | 112         | 84         | GH200 (-25%)      | 2D                    |
| map_0                         | 50          | 34         | GH200 (-32%)      | 2D, C2E gather        |
| map_31                        | 33          | 27         | GH200 (-18%)      | 2D                    |
| **map_85** (forward scan, 1D) | **57**      | **63**     | **MI300A (-10%)** | wavefront-aligned win |
| map_90 (back-sub scan, 1D)    | 27          | 24         | GH200 (-11%)      |                       |
| map_91 (Rayleigh damping)     | 7           | 4          | GH200             | small                 |
| map_13 (boundary)             | 6           | 5          | GH200             | small                 |
| map_100_0 (1D split)          | 6           | 6          | tie               |                       |
| map_111_0 (1D split)          | 5           | 6          | MI300A            | small                 |
| map_35 (zeroing)              | 4           | 2.5        | GH200             | small                 |

**MI300A wins on:** the forward scan (map_85, -10%) and one trivial 1D split (map_111_0).
GH200 wins on every other kernel — most by 18-32%, consistent with its higher
HBM peak and saturation. The forward-scan win confirms our hypothesis that
64-thread blocks fit AMD's 64-wide wavefronts perfectly (1 wave = 1 block,
no scheduling overhead between waves), while NVIDIA has to manage 2 warps per
block on the same kernel.

Sources: MI300A from rocprofv3 kernel-trace; GH200 from `gh200_all_kernels.ncu-rep`
(`gpu__time_duration.sum`, single-iteration ncu run).

## Key Findings

### 1. DaCe-generated kernels are bandwidth-bound, with a coalescing gap

Numbers for map_100_fieldop_1 at `(256,1,1)`:

- **Demand bytes**: 720 MB per invocation (228 B/thread × 3,156,224 threads, re-counted from GCN assembly).
- **HBM moved**: 454 MB (rocprof-compute analyze §4.1.9: 2.43 TB/s × 187 μs).
- **Cache absorption (vL1D + L2 + Infinity Cache combined)**: 37% of demand.
- **HBM utilization**: 2.43 TB/s = **70% of 3.47 TB/s peak**.
- **vL1D Coalescing**: only **27% of peak** ⚠️ (per-CU L1 issues many requests per element due to scattered C2E gather).

Detailed cache-hierarchy breakdown and provenance in [Deep Analysis](docs/DEEP_ANALYSIS.md).

### 2. The MI300A vs GH200 gap shrinks substantially with `(256,1,1)`

- The original 1.4-2.1x per-kernel gap from AMD_INTRODUCTION.md was largely a block
  size mismatch — the DaCe default `(32,8)` is NVIDIA-friendly but wrong for AMD.
- After setting `gpu_block_size_2d=(256,1,1)` **on MI300A**, the gap vs GH200-default shrinks from 1.38x → 1.08x.
- With `(256,1,1)` **on both platforms** (now verified optimal for both), the gap is **1.15x** — closer to the HBM-peak hardware ratio (0.87 → 1.15 reciprocal).
- For map_100_fieldop_1 specifically: GH200 at 89% of 4 TB/s HBM peak, MI300A at 70%
  of 3.47 TB/s HBM peak. GH200 is closer to its HBM ceiling.

### 3. Inter-kernel overhead is small (verified A/B)

| Platform | Block size      | Wall first→last | Σ kernel times | Gap        | Gap %     |
| -------- | --------------- | --------------- | -------------- | ---------- | --------- |
| MI300A   | (32,8) baseline | 1316 μs         | 1314 μs        | **6.3 μs** | **0.47%** |
| MI300A   | (256,1,1)       | 847 μs          | 840 μs         | **9.1 μs** | **1.08%** |
| GH200    | (256,1,1)       | 491 μs          | 474 μs         | 17.2 μs    | 3.49%     |

The previously reported "86 μs (10%) inter-kernel gap" was a calculation artifact
(rocprofv3 GPU kernel time vs pytest-benchmark wall time, mixing GPU and host overhead).

**Kernel fusion would save at most ~10-20 μs per solver call. Not pursued.**

### 4. Occupancy is NOT the bottleneck

- Already at max occupancy (8 waves/SIMD) for 10 of 12 kernels
- Forcing more waves causes 2.5x slowdown due to register spilling

### 5. C2E scatter — investigated, NOT the wall-clock bottleneck (verified 2026-04-20)

- vL1D coalescing is **27% of peak** on map_100_fieldop_1 (MI300A). This means each
  vL1D request fetches more sectors than ideal — 3-4× wasteful at the cache layer.
- **BUT vL1D Bandwidth Utilization is only 27.79%** — the cache has tons of spare
  capacity. The 27% inefficiency wastes a metric we have plenty of.
- **The kernel is HBM-bound (70% of HBM peak)**, not vL1D-bound. Improving vL1D
  coalescing doesn't help wall-clock unless it reduces L2/HBM traffic.
- **L1 hit rate is 72%** — only 28% of vL1D requests actually go to L2. Even fixing
  C2E gather coalescing perfectly (27% → 100%) would reduce L2 traffic by a small
  fraction of that 28%.
- **Predicted wall-clock impact of edge reordering: \<2%, possibly less.** Real
  bottleneck is HBM traffic — see opportunity #2 (intermediate fusion) instead.
- See [Deep Analysis: C2E scatter](docs/DEEP_ANALYSIS.md#c2e-scatter-analysis)
  for measurement detail and reproduction commands.

### 6. L2 cache behavior depends on block size (MI300A)

- With baseline `(32,8)`: ~15-20% hit rate (heavy 2D kernels) — streaming pattern.
- With `(256,1,1)`: **L2 hit rate jumps to 32-50%** on heavy 2D kernels because
  cell-consecutive threads on the same CU share cache lines. Main driver of the 20% speedup.
- 1D / scan kernels are unaffected.

### 7. AMD per-CU L1 dramatically out-caches NVIDIA per-SM L1 on this kernel

For map_100_fieldop_1 at each platform's matching `(256,1,1)`:

| Cache layer          | MI300A (vL1D)        | GH200 (L1)             | Ratio                   |
| -------------------- | -------------------- | ---------------------- | ----------------------- |
| Per-cache size       | 32 KB per CU         | 256 KB per SM          | NVIDIA has 8× per-cache |
| Total L1 across chip | 228 × 32 KB = 7.3 MB | 132 × 256 KB = 33.8 MB | NVIDIA has 4.6× total   |
| **L1 hit rate**      | **72.4%**            | **13.1%**              | **MI300A wins 5.5×**    |
| **L2 hit rate**      | 47.5%                | 48.6%                  | tie                     |

NVIDIA has more total L1 storage AND a bigger per-cache size, but on this kernel
**MI300A's smaller, more-numerous caches catch dramatically more reuse at L1**.
Interestingly, **L2 hit rates are nearly identical (~48% on both)** — so the
overall cache hierarchy ends up catching similar fractions of demand traffic
on both architectures, but MI300A absorbs more at L1, GH200 absorbs more at L2.

**Why MI300A's L1 hits 5× more (plausible hypotheses — not pinned down):**

Both architectures run multiple blocks concurrently per CU/SM, so "per-CU isolation"
is too simple. Possible contributing factors:

1. AMD's wavefront is 64 wide vs NVIDIA's 32-wide warp, so `(256,1,1)` = 4 waves
   on AMD vs 8 warps on NVIDIA on the same block. Less scheduling overhead and
   potentially fewer concurrent working sets competing per cache slice.
2. NVIDIA's 256 KB L1 is shared with shared memory + texture cache; effective
   L1-data portion is smaller than the 256 KB headline.
3. AMD's vL1D may have different replacement policy / faster refill latency
   that suits this access pattern better.
4. GH200 (32,8) has **higher** L1 hit rate (36.5%) than (256,1,1) (13.1%) — smaller
   blocks → fewer concurrent threads per SM → less cache pressure. This at least
   confirms cache contention is a factor on NVIDIA.

**We haven't isolated which factor dominates** — pinning it down would require
microbenchmarks (different cache replacement policies, refill latencies, working
set fits) outside what these profilers expose. The profile tools tell us *what*
(72% vs 13%), not *why*.

**Counter-intuitive but verified (mechanism now proven 2026-04-20):** GH200
(256,1,1) is faster overall (-5%) despite WORSE L1 hit rate. **L2 picks up
the slack**:

| Block     | L1 hit | L2 read hit | L2 overall hit | DRAM bytes | Duration |
| --------- | ------ | ----------- | -------------- | ---------- | -------- |
| (32,8)    | 36.5%  | 14.7%       | 34.9%          | 432 MB     | 123 μs   |
| (256,1,1) | 13.1%  | **36.5%**   | **48.6%**      | 413 MB     | 117 μs   |

When L1 thrashes at the larger block size, GH200's 50 MB L2 catches the
displaced working set. Net DRAM traffic drops 4%, duration drops 5%. L2 read
hit jumps 2.5× (14.7% → 36.5%) — the cache hierarchy works exactly as designed:
data the L1 can't hold falls into the much larger L2.

Source: `gh200_m100_l2_256x1.ncu-rep` and `gh200_m100_l2_32x8.ncu-rep`,
metrics `lts__t_sectors_op_read_lookup_hit.sum` and `_miss.sum`. The aggregate
`lts__t_sectors_lookup_hit_rate.pct` returned `n/a` (NVIDIA metric quirk),
so we computed hit/(hit+miss) from the raw counters.

**Implication:** AMD's many-small-caches design is not just different from
NVIDIA's; it's actively better-suited for kernels with localized cell-block
working sets. This is a structural advantage that block-size tuning can't
replicate on NVIDIA.

## Methodology notes

- **A/B testing:** Same node, same gt4py version, same cache state, same number of
  runs; one variable changes per comparison.
- **Two different timers** appear in this document:
  - **pytest-benchmark** (Python wall time): includes GT4Py dispatch overhead.
    Used in early experiments; absolute values run higher.
  - **GT4Py Timer** (`GT4PY_COLLECT_METRICS_LEVEL=10`): C++ level, closer to the
    actual kernel launches. More accurate. **Use this for all new measurements.**
    Results from different timers are not directly comparable.

## Tooling Notes

### Roofline generation

The original `benchmark_solver.sh` could not generate roofline plots — `rocprof-compute`
concatenates all kernel names into the PDF filename, exceeding the 255-char filesystem limit.
Fixed in `benchmark_solver_roofline.sh` by profiling one kernel at a time in a loop.

### rocpd segfault

`--format-rocprof-output rocpd` causes segfaults (noted in original benchmark_solver.sh TODO).
Workaround: omit this flag, use default CSV format.

### Bandwidth measurement methodology

**Important:** Do not use `TCC_MISS × cache_line_size` to estimate bandwidth. This
counts L2-to-HBM traffic at cache line granularity, which undercounts total data
moved (misses L2 hits) and overcounts (each miss loads a full 64B line even if only
8B is used). Instead, count `global_load`/`global_store` instructions from the GCN
assembly (`hipcc -S`) and multiply by their width × thread count.

**`extract_pmc.py` cache plane formulas** were finalized 2026-05-03. The
script now reports three columns measuring distinct cache planes:

- `L1↔L2(TB/s)` = `(TCC_READ_sum + TCC_WRITE_sum) × 64 / dur`. Upstream plane.
- `L2Fab(TB/s)` = rocprof-compute §2.1.23 + §2.1.24 formula
  (`128×BUBBLE + 64×(RDREQ−BUBBLE−RDREQ_32B) + 32×RDREQ_32B` for reads,
  `64×WRREQ_64B + 32×(WRREQ−WRREQ_64B)` for writes+atomics). Downstream plane.
- `HBM(TB/s)` = `L2Fab × DRAM-traffic-fraction`. Validated against
  rocprof-compute analyze §4.1.9 within 0.4% on map_100_fieldop_1_0_0
  (nid002510): 2.59 vs 2.601 TB/s.

Earlier (2026-04-19) the script reported `(TCC_READ + TCC_WRITE) × 64`
mis-labeled as HBM; it's L2-side throughput, kept as the L1↔L2 column now.
Earlier-still (pre-2026-04-19) `(TCC_EA0_RDREQ + TCC_EA0_WRREQ) × 64` was
wrong by 1.62× because EA0 counters are channel-0 only.

## Files

- `amd_scripts/benchmark_solver.sh` — original benchmark + trace + rocprof-compute
- `amd_scripts/benchmark_solver_roofline.sh` — per-kernel roofline generation (fixed filename issue)
- `amd_scripts/benchmark_solver_occupancy.sh` — occupancy sweep via source-level attribute patching
- `amd_scripts/bw_benchmark.hip` — synthetic HIP bandwidth benchmark matching kernel patterns
- `amd_scripts/analyze_c2e_reorder.py` — C2E scatter analysis and edge reordering test
- `amd_scripts/set_waves_per_eu.py` — patch DaCe-generated HIP with waves_per_eu attribute
- `amd_scripts/patch_cse.py` — patch redundant reads (confirmed unnecessary)
- `amd_scripts/patch_lds_c2e_v2.py` — stage C2E gather through LDS (neutral)
- `amd_scripts/extract_pmc.py` — extract per-kernel duration, L1↔L2, L2-Fabric, HBM BW, L2 hit rate from rocprof-compute pmc_perf.csv
- `workloads/rcu_amd_*/` — per-kernel profiling data (rocprof-compute output)
- `pmc_perf.csv` — hardware counter data for all kernels

______________________________________________________________________

**For full evidence and detailed measurements:**

- [docs/HARDWARE_REFERENCE.md](docs/HARDWARE_REFERENCE.md)
- [docs/DEEP_ANALYSIS.md](docs/DEEP_ANALYSIS.md)
- [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md)
