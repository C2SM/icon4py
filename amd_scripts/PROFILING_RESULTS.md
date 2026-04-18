# MI300A Profiling Results — vertically_implicit_solver_at_predictor_step

Date: 2026-03-30 (updated with corrected bandwidth analysis)

## Summary

Profiled the `vertically_implicit_solver_at_predictor_step` GT4Py program on MI300A (Beverin, gfx942, ROCm 7.1.0).
Baseline median runtime: **~820 μs** (pytest-benchmark timer, includes Python/GT4Py call overhead).

**Note on timers:** Results in this document use two different timers:
- **pytest-benchmark**: measures the full Python function call including GT4Py dispatch overhead.
  Used for: baseline (0.820ms), block size tuning, loop blocking experiments.
- **GT4Py Timer** (`GT4PY_COLLECT_METRICS_LEVEL=10`): measures closer to the actual kernel
  launches (C++ level). More accurate for comparing raw GPU performance.
  Used for: fuse_tasklets A/B test, GH200 comparison.
Results from different timers should not be compared directly. The GT4Py Timer typically
reports lower values than pytest-benchmark (e.g. 0.837ms vs 0.820ms for the same baseline).
Going forward, all new measurements should use the GT4Py Timer.

**Key finding: individual kernels achieve ~94% of HBM peak bandwidth.** The earlier 40.5% estimate
was incorrect — it used TCC_MISS-based traffic (which undercounts) and an outdated kernel time.
The correct analysis shows the DaCe-generated code is performing well. The MI300A vs GH200 gap
is due to hardware bandwidth differences, not software inefficiency.

## Per-Kernel Timing (rocprofv3 kernel-trace, median ns)

| Kernel | Time (μs) | Stencil |
|--------|-----------|---------|
| map_100_1 | **207** | thermo results + dwdz (split 1) |
| map_115_1 | **183** | explicit rho/exner + divergence (split 1) |
| map_60 | **133** | solver coefficients + w explicit |
| map_0 | 59 | contravariant correction (C2E gather) |
| map_85 | 58 | tridiag forward sweep (scan) |
| map_31 | 41 | w explicit term |
| map_90 | 25 | tridiag back-sub (scan) |
| map_91 | 9 | Rayleigh damping |
| map_100_0 | 6 | thermo results + dwdz (split 0, K=0 only) |
| map_115_0 | 5 | explicit rho/exner + divergence (split 0, K=0 only) |
| map_13 | 6 | contravariant correction (lower boundary) |
| map_35 | 4 | zeroing temporaries |
| **Total kernels** | **734** | |
| **Total measured** | **~820** | includes inter-kernel overhead (~86 μs, 10%) |

## Deep Analysis: map_100_fieldop_1 (hottest kernel, 207 μs)

### Bandwidth analysis (corrected)

From the compiled GCN assembly (hipcc -O3 -S):
- 23 × `global_load_dwordx2` (8 bytes each) = 184 bytes/thread
- 3 × `global_load_dword` (4 bytes, C2E int32 indices) = 12 bytes/thread
- 4 × `global_store_dwordx2` (8 bytes each) = 32 bytes/thread
- **Total: 228 bytes/thread**

Total traffic: 228 bytes × 3,143,252 threads = **717 MB**

| Metric | Value |
|--------|-------|
| Kernel time (rocprofv3) | **207 μs** |
| Total data moved per invocation | **717 MB** |
| **Achieved HBM BW** | **3462 GB/s (94.4% of 3668 GB/s peak)** |
| GCN assembly instructions (map_100_1) | 510 |
| GCN assembly instructions (synthetic benchmark) | 430 |

### Synthetic bandwidth benchmark comparison

Hand-optimized HIP kernels matching the solver's access pattern achieve 80-93% of HBM peak.
The DaCe-generated kernel achieves **94.4%** — matching or exceeding hand-written code.
**There is no codegen inefficiency on MI300A.** See appendix for detailed benchmark table.

### Memory access pattern

Per thread: 23 double reads (184 bytes) + 3 int32 C2E indices (12 bytes) +
4 double writes (32 bytes) = 228 bytes total.

- 5 arrays read at both [cell,k] and [cell,k+1]: vertical stencil pattern
- 6 arrays at [cell,k] only: coalesced
- 1 array at [cell] only: broadcast across K
- 3 C2E indirect reads of mass_flux via connectivity table
- 3 geofac_div reads (cell-only, 2D)
- 4 output arrays written at [cell,k]

### C2E scatter analysis

Measured C2E edge index spread for 32 consecutive cells (one warp):

| Metric | Original ordering | Reordered (min-cell) |
|--------|-------------------|----------------------|
| Median spread | 39,881 | 3,612 |
| Cache lines per warp | 14 (mean) | 13.4 (mean) |
| Cache line utilization | **85.7%** | 89.4% |
| Bandwidth amplification | 1.17x | 1.12x |

Despite edge indices spanning 30K-40K range, neighboring cells **share edges** on the
icosahedral grid, so the 96 loads per warp (32 cells × 3 neighbors) hit only ~14 cache lines.
Edge reordering saves ~3.6 MB — not worth pursuing.

### LDS staging experiment for C2E gather

Tested staging the C2E `mass_flux` gather through LDS (shared memory) in the hottest
kernel (`map_100_fieldop_1`). Patched the generated HIP code with `amd_scripts/patch_lds_c2e_v2.py`.

Result: **neutral** (0.614→0.614 ms mean, 0.607→0.609 ms median, 1000 runs).

The C2E gather pattern in the generated code looks like this:

```cpp
for (i = 0; i < 3; i++) {
    int edge_idx = gt_conn_C2E[stride*i + cell_idx];     // load edge index
    double val = mass_flux[edge_idx, K];                  // load value at edge
    __map_fusion_gtir_tmp_29[i] = val;                    // store in register
}
```

Each thread (`cell_idx`) reads its own 3 edge values into a per-thread register array.
The values loaded by thread N are completely different from thread N+1 — there is no
inter-thread data sharing within the gather itself.

Staging through LDS adds: HBM → register → LDS (write) → `__syncthreads()` → LDS (read) → register.
Same HBM traffic, plus extra LDS round-trip. No benefit because there is no data reuse to exploit.

**Where LDS would actually help (not implemented):**

1. **Wavefront-level edge deduplication** — neighboring cells share edges (85.7% cache line
   utilization confirms this). A custom kernel could:
   - Compare edge indices across lanes via `__shfl`
   - Have each unique edge loaded by exactly one thread
   - Distribute the loaded values back via `__shfl` or LDS
   This would reduce HBM traffic, not just stage it. Requires a hand-written HIP kernel,
   not a simple patch.

2. **Cell-broadcast values** — `geofac_div[cell]` is read 3 times per cell. Currently each
   read goes to L1. Staging it once in LDS would save 2 L1 lookups per thread. Marginal.

3. **Tridiagonal scan kernels** (`map_85`, `map_90`) — could benefit from LDS-based block-wide
   scan implementations. Requires algorithmic restructuring, not a patch.

**Conclusion:** LDS staging the existing access pattern doesn't help. Real gains require either
algorithm changes (deduplication via shuffle) or memory layout changes. None are simple patches.

### L2 cache analysis

| Kernel | TCC_MISS | TCC_HIT | L2 Hit Rate |
|--------|----------|---------|-------------|
| map_100_1 | 58334 | 11402 | 16.4% |
| map_115_1 | 54419 | 10625 | 16.3% |
| map_60 | 36256 | 7122 | 16.4% |
| map_31 (no C2E) | 12466 | 1837 | 12.7% |
| map_0 (with C2E) | 17730 | 4326 | 19.8% |

All kernels have similar ~16% L2 hit rate **regardless of C2E usage**. The low hit rate is
due to the streaming access pattern (each element read once), not cache pollution.
Per-L2-slice distribution is perfectly balanced across all 96 slices.

## Per-Kernel Register & Occupancy Profile

From `pmc_perf.csv` (rocprof-compute hardware counters):

| Kernel | Arch VGPR | Accum VGPR | SGPR | LDS | Scratch | Waves/SIMD |
|--------|-----------|------------|------|-----|---------|------------|
| map_100_1 (207 μs) | 56 | 0 | 96 | 0 | 0 | 8 (max) |
| map_115_1 (183 μs) | 48 | 0 | 96 | 0 | 0 | 8 (max) |
| map_60 (133 μs) | 36 | 4 | 64 | 0 | 0 | 8 (max) |
| map_0 (59 μs) | 32 | 0 | 32 | 0 | 0 | 8 (max) |
| map_31 (41 μs) | 12 | 4 | 32 | 0 | 0 | 8 (max) |
| map_85 (58 μs) | 68 | 4 | 32 | 0 | 0 | 7 |
| map_90 (25 μs) | 92 | 4 | 16 | 0 | 0 | 5 |
| map_91 (9 μs) | 8 | 0 | 32 | 0 | 0 | 8 (max) |
| map_100_0 (6 μs) | 56 | 0 | 96 | 0 | 0 | 8 (max) |
| map_115_0 (5 μs) | 48 | 0 | 96 | 0 | 0 | 8 (max) |
| map_13 (6 μs) | 44 | 4 | 32 | 0 | 0 | 8 (max) |
| map_35 (4 μs) | 8 | 0 | 32 | 0 | 0 | 8 (max) |

Waves/SIMD = min(8, floor(512 / Arch_VGPR)). Hardware max is 8 waves per SIMD on gfx942.
10 of 12 kernels are already at max occupancy.

## Occupancy Experiment

Tested the AMD equivalent of NVIDIA's `maxreg` trick by patching DaCe-generated HIP source
with `__attribute__((amdgpu_waves_per_eu(1, N)))` to force higher occupancy.

### What was tried
- ROCm 7.1.0 / clang 20.0.0 does not expose VGPR-limit flags via `-mllvm`:
  - `-mllvm -amdgpu-num-vgpr=N` → `Unknown command line argument`
  - `-mllvm --amdgpu-waves-per-eu=N` → `Unknown command line argument`
  - `HIPFLAGS` with these flags breaks CMake's HIP compiler test (host-side clang rejects GPU flags)
- Source-level `__attribute__((amdgpu_waves_per_eu(min, max)))` compiles successfully
- Created `amd_scripts/set_waves_per_eu.py` to patch the attribute into DaCe-generated .cpp files
- Swept max_waves = 2, 3, 4, 5, 8 via `amd_scripts/benchmark_solver_occupancy.sh`

### Results

| Setting | Median runtime | vs Baseline |
|---------|---------------|-------------|
| Baseline (no attribute) | **0.82 ms** | — |
| waves_per_eu(1, 2) | 2.1 ms | **2.5x slower** |
| waves_per_eu(1, 4) | 2.1 ms | **2.5x slower** |
| waves_per_eu(1, 8) | 2.1 ms | **2.5x slower** |

All non-baseline settings cause ~2.5x regression due to register spilling.

### Why maxreg worked on GH200 but not MI300A
- **GH200 (NVIDIA SM90):** `nvcc` used 128-200+ registers per thread → low occupancy.
  `maxreg` forced fewer registers → more blocks/SM → significant occupancy improvement.
- **MI300A (gfx942):** `hipcc/clang` already uses few VGPRs (32-92) → 5-8 waves/SIMD
  (hardware max is 8). 10 of 12 kernels already at max occupancy.

## Key Findings

### 1. DaCe-generated kernels achieve ~94% of HBM peak bandwidth
- The initial 40.5% estimate was **incorrect** — it used TCC_MISS-based traffic counting
  (356 MB, which only counts L2 cache misses × cache line size) and an outdated kernel time.
- Correct calculation from assembly instruction counts: 228 bytes/thread × 3.14M threads = 717 MB.
  At 207 μs → **3462 GB/s = 94.4% of peak**.
- Synthetic benchmarks with identical access patterns confirm: MI300A delivers 80-93% of peak
  for this workload. The DaCe code is not leaving performance on the table.

### 2. The MI300A vs GH200 gap is hardware, not software
- MI300A achieves ~94% of its 3668 GB/s peak = ~3450 GB/s effective
- GH200 achieves similar efficiency of its ~4000 GB/s peak
- The 1.4-2.1x per-kernel gap from AMD_INTRODUCTION.md is primarily due to:
  - Different peak bandwidths and clock rates
  - Different kernel launch overhead characteristics
  - Different memory controller latency for first-access patterns

### 3. Inter-kernel overhead is the main optimization target
- Total kernel time: 734 μs
- Total measured: ~820 μs
- Inter-kernel gap: **~86 μs (10.5% overhead)**
- This is where kernel fusion and `fuse_tasklets` (~7% on gt4py 1.1.4, ~1.5% on newer versions) help

### 4. Occupancy is NOT the bottleneck
- Already at max occupancy (8 waves/SIMD) for 10 of 12 kernels
- Forcing more waves causes 2.5x slowdown due to register spilling

### 5. C2E scatter is NOT a bottleneck
- Despite edge indices spanning 30K-40K range, neighboring cells share edges
- Cache line utilization: 85.7% (only 1.17x amplification)
- Edge reordering provides negligible improvement (85.7% → 89.4%)

### 6. L2 cache behavior is consistent and expected
- ~16% hit rate across all kernels (with or without C2E)
- This is expected for streaming workloads where each element is read once
- L2 absorbs reuse from k+1 overlaps and cell-only broadcasts (0.78x amplification)

## DaCe Fusion Analysis

### What was tried
- `MapFusion`: applied 9 times (inner tlet maps), but cannot fuse the main maps because
  intermediates (`gtir_tmp_83`, `_96`, `_97`) have multiple consumers
- `SubgraphFusion`: cannot apply — "nodes between maps with incoming edges from outside"
- Root cause: `concat_where` + `broadcast` in GT4Py field operators creates K-domain splits
  (`_0` for K=0, `_1` for K=1..79), producing multiple producers per intermediate

### `fuse_tasklets` optimization
- Added `fuse_tasklets=True` to `model_options.py` for the solver stencil
- This fuses tasklet operations within existing map scopes

Clean A/B test (GT4Py Timer, 30 runs, `amd_profiling` branch, gt4py 1.1.4):

| Config | Mean | Median | StdDev |
|--------|------|--------|--------|
| Without fuse_tasklets | 0.798 ms | 0.797 ms | 0.031 ms |
| With fuse_tasklets | 0.741 ms | 0.732 ms | 0.026 ms |
| **Improvement** | **~7%** | **~7%** | |

Note: Edoardo measured a smaller improvement (~1.5%, 0.782→0.770 ms) on a newer
gt4py/icon4py version. The newer optimization pipeline may already fuse more by default,
leaving less for `fuse_tasklets` to do.

### CSE (Common Subexpression Elimination)
- Detected 28 redundant global loads across all 12 kernels
- Patched generated code to cache duplicate reads
- Result: **no improvement** — compiler already optimizes these at -O3

## GH200 vs MI300A Synthetic Bandwidth Comparison

Same synthetic kernels run on both platforms:
- **Simple stream**: MI300A wins (0.016 vs 0.017 ms) — competitive raw HBM BW
- **Multi-array patterns**: GH200 is consistently **1.23-1.33x** faster
- The gap is hardware-level: GH200's memory controller handles many concurrent streams
  more efficiently

Block size sweep on MI300A synthetic benchmark confirmed that `256×1` is **19% faster**
than the DaCe default `32×8`, which motivated the `gpu_block_size=(256,1,1)` change
in icon4py.

See appendix for detailed benchmark tables.

## MI300A vs GH200 Solver Comparison (GT4Py Timer)

Date: 2026-04-17

Using the GT4Py Timer (C++ level, closest to kernel launches) for apples-to-apples comparison.
Both on `amd_profiling_staging_main` gt4py, `amd_profiling_main` icon4py, regional grid.

| Platform | Config | Mean | Median | Runs |
|----------|--------|------|--------|------|
| MI300A | baseline (32,8,1) | 0.768 ms | **0.763 ms** | 1000 |
| MI300A | fuse_tasklets only | 0.763 ms | 0.760 ms | 30 |
| MI300A | **(256,1,1) only** | **0.618 ms** | **0.612 ms** | **1000** |
| MI300A | (256,1,1) + fuse_tasklets | 0.640 ms | 0.632 ms | 30 |
| MI300A | (256,1,1) + blocking (threshold=3) | 0.650 ms | 0.649 ms | 30 |
| GH200 | defaults (32,8,1) | 0.559 ms | **0.559 ms** | 1000 |

| Comparison | Ratio |
|------------|-------|
| GH200 vs MI300A baseline | GH200 **1.36x** faster |
| GH200 vs MI300A best (256,1,1) | GH200 **1.09x** faster |

The performance gap narrowed from **1.36x to 1.09x** with (256,1,1) block size on MI300A.
The (256,1,1) block size gives a **~20% improvement** over the default (32,8,1).

Note: fuse_tasklets is neutral on this gt4py version (`amd_profiling_staging_main`).
On the older gt4py 1.1.4 (`amd_profiling_staging`) it helped ~7%. The (256,1,1)
block size is the only optimization that matters on the current version.

## GPU Block Size Tuning

Date: 2026-04-14

### (256,1,1) block size for all maps

Setting `gpu_block_size=(256,1,1)` for all 1D and 2D maps on MI300A gives a significant
improvement by putting all threads on the Cell (horizontal) dimension for maximum coalescing.

```python
# In model_options.py, ROCM device block:
optimization_args["gpu_block_size"] = (256, 1, 1)
optimization_args["gpu_block_size_1d"] = (256, 1, 1)
optimization_args["gpu_block_size_2d"] = (256, 1, 1)
```

### Results (GT4Py Timer, `amd_profiling_staging_main` gt4py)

| Config | Mean | Median | Runs | vs Baseline |
|--------|------|--------|------|-------------|
| Baseline (32,8,1) | 0.768 ms | 0.763 ms | 1000 | — |
| fuse_tasklets only | 0.763 ms | 0.760 ms | 30 | neutral |
| **(256,1,1) only** | **0.618 ms** | **0.612 ms** | **1000** | **-19.8%** |
| (256,1,1) + fuse_tasklets | 0.640 ms | 0.632 ms | 30 | -17.2% |
| (256,1,1) + blocking (threshold=3) | 0.650 ms | 0.649 ms | 30 | -15.0% |
| (256,1,1) + blocking (threshold=3) | 0.649 ms | -14.4% |

Baseline (GT4Py Timer, no optimizations) measurement pending.

Earlier pytest-benchmark results for reference (different timer, different gt4py version):

| Config | Block size | pytest-benchmark Median | vs Baseline |
|--------|-----------|------------------------|-------------|
| Baseline | (32,8,1) default | 0.820 ms | — |
| (256,1,1) all maps | (256,1,1) | 0.703 ms | -14.3% |
| (64,6,1) all maps | (64,6,1) | 0.756 ms | -7.8% |

### Why (256,1,1) is faster

- All 256 threads in a block process consecutive cells → perfect coalescing for `array[cell, K]`
- The default (32,8,1) spreads 8 threads across K levels, which wastes coalescing potential
  since K is the non-unit-stride dimension
- MI300A wavefronts are 64 wide, so 256 = 4 wavefronts per block — good occupancy

## Loop Blocking (K-Blocking) Experiment

Date: 2026-04-14

### What is loop blocking

Tiles the K dimension into blocks of `blocking_size`. Computations that don't depend on K
(e.g. connectivity reads, geofac_div, cell-only arrays) are moved outside the inner K-loop,
so they execute once per block instead of once per K-level.

Based on GT4Py PR: https://github.com/GridTools/gt4py/compare/main...iomaganaris:gt4py:extend_loopblocking

### Results (pytest-benchmark median — needs re-measurement with GT4Py Timer)

| Config | Median | vs Baseline |
|--------|--------|-------------|
| Baseline (32,8) | 0.820 ms | — |
| Blocking only (32,8 overwritten) | 0.797 ms | -2.8% |
| Blocking + (256,1,1) blocked map only | 0.700 ms | -14.6% |
| (256,1,1) all maps, no blocking | 0.703 ms | -14.3% |
| (256,1,1) all maps + blocking threshold=3 | 0.716 ms | -12.7% |
| (256,1,1) all maps + blocking threshold=1 | 0.730 ms | -11.0% |
| (256,1,1) all maps + blocking threshold=0 | 0.723 ms | -11.8% |

### Analysis

- Loop blocking by itself gives a 2.8% speedup by reducing redundant K-independent work
  in map_0 (the only kernel that met `independent_node_threshold=3`)
- The block size change from (32,8) to (256,1,1) is responsible for most of the improvement
  (~12% out of the 14.6% total)
- When (256,1,1) is already set globally on all maps, enabling loop blocking on top actually
  makes things slightly worse (0.703→0.716ms). The extra overhead from the coarse loop
  (bounds checks, `min()` calls) outweighs the savings, since the coalescing improvement
  — which was the main driver — is already captured by the global block size setting
- We also tried lowering `independent_node_threshold` to 1 and 0 to block more kernels,
  but this degraded performance further
- **Bottom line: setting (256,1,1) on all maps is the best approach for MI300A (14.3%).
  Loop blocking gives 2.8% on its own but becomes redundant once (256,1,1) is applied
  globally**

### Bugs found and fixed in GT4Py transformation pipeline

1. **GPU transformation resets block size:** `sdfg.apply_gpu_transformations()` in
   `_gt_auto_configure_maps_and_strides` creates new GPU-scheduled maps that don't inherit
   `gpu_block_size` set by `LoopBlocking.apply()`. Fixed by adding a post-GPU override that
   finds maps with `__gtx_coarse_` parameter and re-applies the configured block size.

2. **`__maxnreg__` is CUDA-only, breaks HIP:** DaCe emits `__maxnreg__(N)` when `gpu_maxnreg>0`,
   which is not valid in HIP/ROCm (causes compilation error). Changed `LoopBlocking` property
   defaults to `None` so nothing is set unless explicitly requested.

3. **Block size was (32,8,1) not (256,1,1):** After blocking, outer map has 2 dims
   (Cell, __gtx_coarse_K). `GPUSetBlockSize` classified it as 2D and applied `(32,8,1)`.
   Fixed by the post-GPU override described above.

### Changes to GT4Py (on top of extend_loopblocking PR)

**loop_blocking.py:**
- Added `gpu_block_size` and `gpu_maxnreg` as configurable properties (default `None`)
- Made `apply()` conditional — only sets when not `None`

**auto_optimize.py:**
- Added `blocking_gpu_block_size` and `blocking_gpu_maxnreg` parameters
- Added post-GPU-transformation override in `_gt_auto_configure_maps_and_strides()`

## Optimization Opportunities

### Confirmed helpful
1. **`fuse_tasklets`** — 7-8% improvement on gt4py 1.1.4 (~1.5% on newer versions per Edoardo's measurements)
2. **GPU block size (256,1,1) for all maps** — 14.3% improvement on MI300A. All threads on
   Cell dimension maximizes coalescing.

### Worth investigating
3. **Grid size divisible by 6** — preliminary data shows additional ~4% improvement (0.617→0.594ms
   on a different gt4py branch). Possibly related to even work distribution across MI300A's
   6 XCDs. Needs further investigation.
4. **Kernel fusion** — reduces 12 kernel launches to fewer, eliminating ~86 μs inter-kernel
   overhead (10%). Blocked by DaCe `SubgraphFusion` limitations.

### Not helpful (confirmed)
5. ~~Loop blocking on MI300A~~ — 2.8% alone, but redundant when (256,1,1) is applied globally; combining both is slower than (256,1,1) alone
6. ~~C2E scatter / edge reordering~~ — C2E is well-localized (85.7% cache utilization)
7. ~~Occupancy tuning~~ — already at max, causes spilling
8. ~~Register limiting via compiler flags~~ — not available in ROCm 7.1.0
9. ~~CSE~~ — compiler handles it
10. ~~Many-array BW limitation~~ — synthetic benchmark proves MI300A handles 20+ arrays at 93% peak
11. ~~Block size (64,6,1)~~ — worse than (256,1,1) by 7%

## Tooling Notes

### Roofline generation
The original `benchmark_solver.sh` could not generate roofline plots — `rocprof-compute` concatenates
all kernel names into the PDF filename, exceeding the 255-char filesystem limit.

Fixed in `benchmark_solver_roofline.sh` by profiling one kernel at a time in a loop. The roofline
PDFs are generated automatically by `rocprof-compute profile` inside
`workloads/<profile_name>/MI300A_A1/` — no special roofline flag needed.

### rocpd segfault
`--format-rocprof-output rocpd` causes segfaults (noted in original benchmark_solver.sh TODO).
Workaround: omit this flag, use default CSV format.

### Bandwidth measurement methodology
**Important:** Do not use `TCC_MISS × cache_line_size` to estimate bandwidth. This counts
L2-to-HBM traffic at cache line granularity, which undercounts total data moved because it
misses L2 hits and overcounts because each miss loads a full 64B line even if only 8B is used.
Instead, count `global_load`/`global_store` instructions from the GCN assembly (`hipcc -S`)
and multiply by their width × thread count.

## Files

- `amd_scripts/benchmark_solver.sh` — original benchmark + trace + rocprof-compute (unchanged)
- `amd_scripts/benchmark_solver_roofline.sh` — per-kernel roofline generation (fixed filename issue)
- `amd_scripts/benchmark_solver_occupancy.sh` — occupancy sweep via source-level attribute patching
- `amd_scripts/bw_benchmark.hip` — synthetic HIP bandwidth benchmark matching kernel patterns
- `amd_scripts/analyze_c2e_reorder.py` — C2E scatter analysis and edge reordering test
- `amd_scripts/set_waves_per_eu.py` — utility to patch DaCe-generated HIP with waves_per_eu attribute
- `amd_scripts/patch_cse.py` — utility to patch redundant reads (confirmed unnecessary)
- `workloads/rcu_amd_profiling_solver_regional_map_*/` — per-kernel profiling data and roofline PDFs
- `pmc_perf.csv` — hardware counter data for all kernels

## Appendix: Detailed Synthetic Benchmark Data

### MI300A synthetic bandwidth benchmarks

Hand-optimized HIP kernels matching the solver's access pattern and grid dimensions.
All run with dim3(1244,10) × dim3(32,8):

| Benchmark | Median (ms) | BW (GB/s) | % of peak |
|-----------|-------------|-----------|-----------|
| 1. Stream (1R + 1W) | 0.016 | 3113 | 84.9% |
| 2. Many arrays flat (20R + 4W) | 0.207 | 2952 | 80.5% |
| 3. Many arrays + k+1 offsets (25R + 4W) | 0.235 | 3105 | 84.7% |
| 4. Full pattern + C2E indirection | 0.216 | 3388 | 92.4% |
| 5. Same as #4 with DaCe grid dims | 0.214 | 3411 | 93.0% |
| 6. DaCe-style (56 args, per-array indexing) | 0.185 | 3269 | 89.1% |
| **DaCe map_100_1 (actual)** | **0.207** | **3462** | **94.4%** |

### MI300A vs GH200 synthetic (dim3(32,8) blocks, 39788 cells × 80 K-levels)

| Benchmark | MI300A (ms) | GH200 (ms) | GH200 speedup |
|-----------|-------------|------------|----------------|
| 1. Stream (1R+1W) | 0.016 | 0.017 | 0.94x (MI300A faster) |
| 2. Many arrays flat (20R+4W) | 0.205 | 0.167 | **1.23x** |
| 3. k+1 offsets (25R+4W) | 0.231 | 0.174 | **1.33x** |
| 4. Full + C2E | 0.212 | 0.166 | **1.28x** |
| 5. DaCe grid | 0.212 | 0.166 | **1.28x** |
| 6. DaCe-style 56 args | 0.183 | 0.145 | **1.26x** |

### Block size sweep on MI300A synthetic benchmark

| Block size | Median (ms) | vs 32×8 |
|-----------|-------------|---------|
| 32×8 (DaCe default) | 0.212 | baseline |
| 64×4 (wavefront-aligned) | 0.190 | 10% faster |
| 128×2 | 0.192 | 9% faster |
| 256×1 | 0.171 | **19% faster** |

Note: DaCe's `gpu_utils.py` overrides `default_block_size` config and always uses its own 2D
tiling heuristic. Setting `DACE_compiler_cuda_default_block_size="256,1,1"` alone shows
**no improvement** because DaCe still generates dim3(32,8) blocks. The real fix is setting
`gpu_block_size` per-dim in `model_options.py` as we did (see main section).
