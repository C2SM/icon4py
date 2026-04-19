# MI300A Profiling Results — vertically_implicit_solver_at_predictor_step

Date: 2026-03-30 (initial), 2026-04-19 (latest update)

## Summary

Profiled the `vertically_implicit_solver_at_predictor_step` GT4Py program on
MI300A (Beverin, gfx942, ROCm 7.1.0).

### Headline result

A one-line config change — `gpu_block_size_2d=(256,1,1)` for ROCm in `model_options.py`
— gives **21% speedup** on this stencil and closes most of the MI300A vs GH200 gap.

| Platform | Config | GT4Py Timer median (1000 runs) |
|----------|--------|-------------------------------|
| MI300A | DaCe default `(32,8,1)` | 0.763 ms |
| MI300A | **`gpu_block_size_2d=(256,1,1)`** | **0.604 ms (-21%)** |
| GH200 | defaults `(32,8,1)` | 0.559 ms |

Gap MI300A → GH200: **1.36x → 1.13x**. The remaining gap is hardware (GH200 has
~10% higher HBM bandwidth and saturates it; MI300A doesn't).

### Why (256,1,1) helps

The default `(32,8)` thread block layout works fine for NVIDIA's 32-wide warps but
mismatches AMD's 64-wide wavefronts. With `(256,1,1)`:
- All 256 threads in a block process consecutive cells → wavefront-aligned access
- Cell-consecutive threads on the same CU dramatically improve cache reuse:
  **L2 hit rate jumps 15-20% → 32-50%** on the heavy 2D kernels (verified)
- vL1D hit rate reaches **72%** on heavy kernels (per-CU TCP catches reuse)
- Per-kernel duration drops 18-24%
- Scan kernels (1D maps) are unchanged

**Important caveat:** `(256,1,1)` aligns access to wavefronts, but **vL1D coalescing
on the heavy kernel is still only 27% of peak** (per-kernel rocprof-compute analyze
on map_100_fieldop_1, §16.1.3). The C2E gather scatters threads across edge memory
even with cell-consecutive blocks — the cell-side accesses coalesce well, but the
edge-side gather (3 edges per cell, non-contiguous indices) does not.

### What does NOT help (verified, with measurements)

| Optimization | Result |
|--------------|--------|
| Loop blocking (K-blocking) | 2.8% on its own, redundant once `(256,1,1)` is applied |
| `fuse_tasklets` | Neutral on current gt4py (was 7% on older 1.1.4) |
| Kernel fusion | Inter-kernel gap is only 7-11 μs (1-2%) — not worth pursuing |
| Forcing higher occupancy | 2.5x slowdown (register spilling); kernels already at max |
| Register limiting via compiler flags | Not exposed in ROCm 7.1.0 |
| CSE (common subexpression elimination) | Compiler handles it at -O3 |
| Block size `(64,6,1)` | 7% worse than `(256,1,1)` |
| LDS staging the C2E gather (synthetic test) | Neutral in the synthetic isolated test |

**Note (revisit):** Edge reordering / C2E scatter was previously dismissed
("85.7% cache line utilization"). The 85.7% measured cache-line **fill rate**, not
**coalescing efficiency**. Per-kernel coalescing is **27% of peak on MI300A**
(§16.1.3) and **86.5% sector util on missed sectors on GH200**. Both numbers say
the C2E gather wastes bandwidth on both architectures. **Edge reordering is now
the highest-priority unverified optimization.**

### Methodology notes

- **A/B testing:** Same node, same gt4py version, same cache state, same number of runs;
  one variable changes per comparison.
- **Two different timers** appear in this document:
  - **pytest-benchmark** (Python wall time): includes GT4Py dispatch overhead.
    Used in early experiments; absolute values run higher.
  - **GT4Py Timer** (`GT4PY_COLLECT_METRICS_LEVEL=10`): C++ level, closer to the
    actual kernel launches. More accurate. **Use this for all new measurements.**
  Results from different timers are not directly comparable.

## MI300A vs GH200 Solver Comparison (GT4Py Timer)

Date: 2026-04-17

Setup: gt4py branch `amd_profiling_staging_main`, icon4py branch `amd_profiling_main`,
regional grid, GT4Py Timer (1000 runs). The headline table in the Summary is reproduced
here with the mean column for completeness:

| Platform | Config | Mean | Median |
|----------|--------|------|--------|
| MI300A | baseline (32,8,1) | 0.768 ms | 0.763 ms |
| MI300A | **`gpu_block_size_2d=(256,1,1)`** | **0.611 ms** | **0.604 ms** |
| GH200 | defaults (32,8,1) | 0.559 ms | 0.559 ms |

`fuse_tasklets` is neutral on this gt4py version (was ~7% on older gt4py 1.1.4).
The `(256,1,1)` block size is the only optimization that matters on the current version.

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

## GPU Block Size Tuning

Date: 2026-04-14

The headline result is in the Summary. This section drills into which specific
`gpu_block_size_*d` setting is responsible.

The heavy kernels (map_100_1, map_111_1, map_60) are 2D maps and account for >70%
of total kernel time. The 1D kernels (boundary: map_13, map_35; scans: map_85,
map_90; K=0-only splits: map_100_0, map_111_0) either run a single wavefront or
have data dependencies that don't benefit from larger thread blocks.

```python
# In model_options.py, ROCM device block:
optimization_args["gpu_block_size_2d"] = (256, 1, 1)
```

### A/B sweep (GT4Py Timer, `amd_profiling_staging_main` gt4py, warm cache, 1000 runs)

| Config | Mean | Median | vs Baseline |
|--------|------|--------|-------------|
| Baseline (32,8,1) | 0.768 ms | 0.763 ms | — |
| `gpu_block_size_1d=(256,1,1)` only | 0.771 ms | 0.767 ms | neutral |
| **`gpu_block_size_2d=(256,1,1)` only** | **0.611 ms** | **0.604 ms** | **-20.8%** |
| All three set | 0.618 ms | 0.612 ms | -19.8% |
| fuse_tasklets only | 0.763 ms | 0.760 ms | neutral (30 runs) |
| (256,1,1) + fuse_tasklets | 0.640 ms | 0.632 ms | -17.2% (30 runs) |
| (256,1,1) + blocking (threshold=3) | 0.650 ms | 0.649 ms | -15.0% (30 runs) |

`gpu_block_size_2d=(256,1,1)` is the only setting needed. The others have no effect
(harmless to set, just adds nothing).

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

## Loop Blocking (K-Blocking) Experiment

Date: 2026-04-14

Loop blocking tiles the K dimension into blocks of `blocking_size`. Computations that
don't depend on K (connectivity reads, geofac_div, cell-only arrays) are moved outside
the inner K-loop, so they execute once per block instead of once per K-level.

Based on GT4Py PR: https://github.com/GridTools/gt4py/compare/main...iomaganaris:gt4py:extend_loopblocking

### Results (pytest-benchmark median — needs re-measurement with GT4Py Timer)

| Config | Median | vs Baseline |
|--------|--------|-------------|
| Baseline (32,8) | 0.820 ms | — |
| Blocking only (32,8 block) | 0.797 ms | -2.8% |
| Blocking + (256,1,1) on blocked map | 0.700 ms | -14.6% |
| (256,1,1) all maps, no blocking | 0.703 ms | -14.3% |
| (256,1,1) + blocking threshold=3 | 0.716 ms | -12.7% (worse) |
| (256,1,1) + blocking threshold=1 | 0.730 ms | -11.0% (worse) |
| (256,1,1) + blocking threshold=0 | 0.723 ms | -11.8% (worse) |

**Bottom line:** Loop blocking gives 2.8% on its own (only `map_0` met
`independent_node_threshold=3`) but becomes redundant once `(256,1,1)` is applied
globally — adding it on top of `(256,1,1)` makes things slightly worse, because the
coarse loop overhead (bounds checks, `min()` calls) outweighs the savings.

### Bugs found and fixed in GT4Py transformation pipeline (kept for reference)

While testing loop blocking we patched three issues in the gt4py transformation
pipeline. These changes live in our local fork only; if loop blocking is revived as
an optimization, they need to be upstreamed:

1. **GPU transformation resets block size:** `sdfg.apply_gpu_transformations()` in
   `_gt_auto_configure_maps_and_strides` creates new GPU-scheduled maps that don't
   inherit `gpu_block_size` set by `LoopBlocking.apply()`. Fixed by a post-GPU
   override that finds maps with `__gtx_coarse_` parameter and re-applies the configured
   block size.
2. **`__maxnreg__` is CUDA-only, breaks HIP:** DaCe emits `__maxnreg__(N)` when
   `gpu_maxnreg>0`, which is not valid in HIP/ROCm (compile error). Changed
   `LoopBlocking` property defaults to `None` so nothing is set unless explicitly
   requested.
3. **Block size was (32,8,1) not (256,1,1):** After blocking, outer map has 2 dims
   (Cell, `__gtx_coarse_K`). `GPUSetBlockSize` classified it as 2D and applied
   `(32,8,1)`. Fixed by the post-GPU override above.

Code-level details: added `gpu_block_size`/`gpu_maxnreg` properties to `loop_blocking.py`
(default `None`, conditional in `apply()`), and `blocking_gpu_block_size`/`blocking_gpu_maxnreg`
parameters to `auto_optimize.py` plus the post-GPU override in `_gt_auto_configure_maps_and_strides()`.

## Optimization Opportunities

### Confirmed helpful
1. **`fuse_tasklets`** — 7% improvement on gt4py 1.1.4 (~1.5% on newer versions per Edoardo's measurements)
2. **GPU block size (256,1,1) for all maps** — 14.3% improvement on MI300A. All threads on
   Cell dimension maximizes coalescing.

### Worth investigating
3. **Edge reordering / C2E gather coalescing** — promoted from "not helpful" after
   per-kernel rocprof-compute analyze showed vL1D coalescing is only **27% of peak**
   on map_100_fieldop_1 (MI300A) and 86.5% sector util on GH200's missed sectors.
   The "85.7% cache utilization" number that previously dismissed this was measuring
   cache-line **fill**, not **coalescing efficiency**. This is now the most likely
   significant code-side optimization.
4. **Grid size divisible by 6** — preliminary data shows additional ~4% improvement (0.617→0.594ms
   on a different gt4py branch). Possibly related to even work distribution across MI300A's
   6 XCDs. Needs further investigation.
5. ~~**Kernel fusion**~~ — verified: actual GPU inter-kernel gap is only 6-11 μs (1-2%)
   on all gt4py versions tested. The earlier "86 μs gap" claim was an incorrect
   subtraction of rocprof kernel-sum from pytest-benchmark wall time. Not worth pursuing.

### Not helpful (confirmed)
6. ~~Loop blocking on MI300A~~ — 2.8% alone, but redundant when (256,1,1) is applied globally; combining both is slower than (256,1,1) alone
7. ~~Occupancy tuning~~ — already at max, causes spilling
8. ~~Register limiting via compiler flags~~ — not available in ROCm 7.1.0
9. ~~CSE~~ — compiler handles it
10. ~~Many-array BW limitation~~ — synthetic benchmark proves MI300A handles 20+ arrays at 93% peak
11. ~~Block size (64,6,1)~~ — worse than (256,1,1) by 7%
12. ~~LDS staging for C2E gather~~ — neutral in the **synthetic isolated test**.
    Should be re-evaluated in-context after edge reordering is tried, since the
    in-stencil gather has different stalls than the synthetic.

## Per-Kernel Timing (rocprofv3 kernel-trace, median μs)

Both columns measured with rocprofv3 kernel-trace, fresh runs on the current gt4py.

| Kernel | Baseline (μs) | (256,1,1) (μs) | Improvement | Stencil |
|--------|---------------|----------------|-------------|---------|
| map_100_1 | 212 | **166** | -22% | thermo results + dwdz (split 1) |
| map_111_1 | 186 | **146** | -22% | explicit rho/exner + divergence (split 1) |
| map_60 | 138 | **112** | -19% | solver coefficients + w explicit |
| map_0 | 60 | 50 | -16% | contravariant correction (C2E gather) |
| map_85 | 58 | 57 | neutral | tridiag forward sweep (scan, 1D) |
| map_31 | 42 | 33 | -21% | w explicit term |
| map_90 | 27 | 27 | neutral | tridiag back-sub (scan, 1D) |
| map_91 | 9 | 7 | -22% | Rayleigh damping |
| map_100_0 | 6 | 6 | neutral | thermo results split 0 (1D) |
| map_111_0 | 5 | 5 | neutral | rho/exner split 0 (1D) |
| map_13 | 6 | 6 | neutral | contravariant correction lower boundary (1D) |
| map_35 | 4 | 4 | neutral | zeroing temporaries (1D) |
| **Total kernels** | **753** | **618** | **-18%** | sum of per-kernel medians |
| **Wall clock (first→last)** | 774 | 660 | -15% | inter-kernel gap: 6-11 μs (1-2%) |
| **GT4Py Timer median** | 760 | 604 | -21% | full SDFG call (1000 runs) |

The 2D heavy kernels (map_100_1, map_111_1, map_60, map_31) all gained 16-20% from
the (256,1,1) block size. 1D kernels and scans are unchanged as expected.

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
| **Demand BW** (assembly-counted) | **3462 GB/s (94.4% of 3668 GB/s peak)** |
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

**Reconciling with per-kernel coalescing measurements:** the 85.7% cache-line
**utilization** here is about *reuse over time* (how much of a fetched cache line
is eventually consumed), and it is genuinely high. But this is independent from
**per-access coalescing** — how efficiently each individual vL1D request retrieves
useful bytes. The per-kernel rocprof-compute analyze on map_100_fieldop_1 shows
**vL1D coalescing is 27% of peak**: each gather access, even if it later contributes
to a well-utilized cache line, requires multiple vL1D requests because the 64
threads in a wavefront target scattered edge addresses.

So edge reordering wouldn't save much *cache memory traffic* (cache lines are
already 85% used), but a different approach — wavefront-level edge deduplication
(`__shfl` or LDS) — could reduce the *number of vL1D requests* per element and
attack the 27% coalescing inefficiency. That's the path worth investigating.

### LDS staging experiment for C2E gather

**Result: neutral** (0.614→0.614 ms mean, 0.607→0.609 ms median, 1000 runs).

Tested staging the C2E `mass_flux` gather through LDS (shared memory) in the hottest
kernel (`map_100_fieldop_1`). Patched the generated HIP code with `amd_scripts/patch_lds_c2e_v2.py`.

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

### Block size effect on MI300A — (256,1,1) only, verified per-kernel

| Kernel | Duration | L2 hit rate |
|--------|----------|-------------|
| map_100_1 | 187 μs | 47.4% |
| map_111_1 | 168 μs | 42.7% |
| map_60 | 125 μs | 42.8% |
| map_0 | 52 μs | 49.8% |
| map_31 | 39 μs | 32.3% |
| map_85 (scan, 1D) | 60 μs | 22.6% |
| map_90 (scan, 1D) | 26 μs | 22.6% |

Source: patched extract_pmc.py on `workloads/rcu_amd_256x1_solver/MI300A_A1/pmc_perf.csv`
(L2 hit from `TCC_HIT_sum/TCC_REQ_sum`, verified to match rocprof-compute analyze §2.1.21
within 0.1% on map_100_fieldop_1).

(Durations from rocprof-compute multi-pass profiling include instrumentation overhead;
clean rocprofv3 numbers are lower.)

**(32,8) baseline column dropped.** Both rocprof-compute workloads on the cluster
(`rcu_amd_256x1_solver/` and `rcu_amd_baseline_solver/`) were actually run with
`Workgroup_Size=256`. There is no true `(32,8)` rocprof-compute run available to
compare against. The previously-quoted (32,8) numbers (15.2% L2 hit rate, 1.65 TB/s
HBM BW, etc.) cannot be re-verified and have been removed. To regenerate: set
`gpu_block_size_2d=(32,8,1)` in `model_options.py`, re-run rocprof-compute, then
re-extract.

### Cross-platform memory hierarchy (MI300A vs GH200) — directly measured

For map_100_fieldop_1 (the hottest kernel), both at their best block size:

| Platform | HBM BW | % of HBM peak | Duration |
|----------|--------|---------------|----------|
| MI300A `(256,1,1)` | **2.43 TB/s** (rocprof-compute §4.1.9) | **70%** of 3.47 TB/s | 187 μs (rocprof-compute) / 166 μs (rocprofv3) |
| GH200 `(32,8)` | 3.55 TB/s (ncu) | 89% of 4 TB/s | 121 μs |

These are the only directly-measured BW numbers. Absorption % (how much L2/L1
caches absorb relative to demand) cannot be computed without a verified
demand-bytes number, which we don't have. The earlier "61% / 43% absorbed"
table was derived from a stale GCN assembly count and a buggy extract_pmc.py;
removed pending re-verification.

(MI300A duration 187 μs is from rocprof-compute multi-pass profiling, which
adds instrumentation overhead; clean rocprofv3 reports 166 μs.)

#### Cross-platform per-kernel comparison (map_100_fieldop_1)

The hottest kernel — 27% of total kernel time.

| Metric | MI300A `(256,1,1)` | GH200 `(32,8)` |
|--------|--------------------|--------------------|
| Duration | 187 μs (rocprof-compute) / 166 μs (rocprofv3) | 121 μs |
| **L2-Fabric / HBM-side BW** | **2.43 TB/s** (70% of 3.47 TB/s peak)<br/>= 1894 Read + 538 Write Gb/s | 3.55 TB/s (89% of 4 TB/s peak) |
| L2 hit rate | **47.5%** (rocprof-compute §2.1.21; matches extract_pmc.py 47%) | _TODO: `lts__t_sectors_lookup_hit_rate.pct`_ |
| **vL1D / L1 hit rate** | **72.41%** (rocprof-compute §2.1.19) | **39.2%** (60.8% L1 sectors miss) |
| **vL1D Coalescing** | **27.03% of peak** (rocprof-compute §16.1.3) ⚠️ | 86.5% sector util on missed sectors |
| **vL1D Stalled on L2 Data** | **48.31%** of cycles (§16.2.0) | n/a equivalent |
| L2-Fabric Read Latency | **1440 cycles** (§2.1.25) | n/a equivalent (GH200 reports L1TEX scoreboard) |
| Top stall reason | **vL1D stalled on L2 data 48%** (memory-latency-bound) | **84.3% L1TEX scoreboard** (29.0/34.4 cycles) |
| Theoretical occupancy | 8 waves/SIMD (max) | **62.5%** (10/16) — limited by **register pressure** |
| IPC | **0.24** (4.87% of peak) | _TODO_ |
| VALU FLOPs % peak | 1.64% | _TODO_ |
| Active threads / wave | 63.97 / 64 (99.95%) | n/a equivalent |

**Provenance:** MI300A from `rocprof-compute analyze -p workloads/rcu_amd_256x1_solver/MI300A_A1/ --dispatch 11 23 35` (averaged across 3 invocations of map_100_fieldop_1). GH200 from `gh200_solver.ncu-rep` opened in ncu UI.

**HBM BW reconciliation:** extract_pmc.py reports 1.50 TB/s / 281 MB; rocprof-compute analyze reports 2.43 TB/s for the same kernel from the same pmc_perf.csv. **The 1.62× discrepancy has not been traced.** Possible causes: extract_pmc.py uses TCC_EA0_RDREQ/WRREQ which may need different aggregation across L2 channels, wrong cache-line scaling, or rocprof-compute counts traffic extract_pmc.py omits. The directly-measured rocprof-compute number (2.43 TB/s, 70% of peak) is what's reported in this table; extract_pmc.py output should not be cited for HBM BW until the discrepancy is traced.

Source for GH200 row: `gh200_solver.ncu-rep` opened in ncu UI; explicit warnings
quoted: "L1TEX scoreboard ... 84.3% of total average of 34.4 cycles between
issuing two instructions"; "theoretical occupancy 62.5% limited by registers";
"only 27.7 of 32 bytes transmitted per sector are utilized ... applies to 60.8%
of sectors missed in L1TEX". This concurs with the prior "83.6% L1TEX scoreboard"
claim (rounding).

Commands to fill the gaps (run on cluster):

```bash
# MI300A — find map_100_fieldop_1 dispatch ID, then:
rocprof-compute analyze -p workloads/rcu_amd_256x1_solver/MI300A_A1/ \
    --dispatch <map_100_id> > /tmp/m100_mi300a.txt

# GH200 — re-run ncu with explicit per-kernel metrics for the missing IPC/L2 cells
ncu --metrics \
    lts__t_sectors_lookup_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
smsp__inst_executed.avg.per_cycle_active \
    -k regex:'map_100_fieldop_1' ...
```

#### What this implies for optimization on map_100_fieldop_1

**MI300A — picture is more nuanced than initial read.** The kernel:
- **Hits HBM at 70% of peak**, not 43% as previously claimed. Closer to bandwidth-bound than absorption-bound.
- **vL1D hit 72%** — good per-CU caching, but...
- **Coalescing only 27% of peak** ⚠️ — vL1D bandwidth efficiency is poor; lots
  of wasted lanes per access. Same root cause as GH200's 86.5% sector util:
  the C2E gather scatters threads across edge memory.
- **vL1D stalls on L2 data 48% of cycles** → memory-latency-bound, not throughput-bound at the vL1D layer.
- IPC 0.24 (4.87% of peak), VALU 1.64% — confirms compute-light, memory-stall-bound.

**Implication change:** On MI300A, the bottleneck is **not** "vL1D already absorbs everything, no headroom." It's:
1. Coalescing inefficiency in vL1D (27% of peak coalescing) wastes ~3-4× the bandwidth needed
2. L2-Fabric read latency 1440 cycles → wave occupancy can't hide it (vL1D stalled on L2 48%)

So **fixing coalescing on the C2E gather should help MI300A too**, contrary to my earlier claim. The mechanism: better coalescing → fewer vL1D requests per element → fewer L2 misses → fewer 1440-cycle stalls.

**GH200 side (HBM 89% saturated, but with leaks):**
- 13.5% sector-utilization waste on the 60.8% of sectors that miss L1
- Likely culprit: same C2E gather (3 edges per cell, stride-y access into edge arrays)
- Register-pressure-limited occupancy (62.5%) → fewer warps to hide the 84% L1TEX scoreboard stalls
- If sector util goes from 86.5% → ~100%, effective HBM demand drops ~13%; kernel could go from 121 μs → ~105 μs (back-of-envelope)

**Concrete things to try (in priority order — each needs A/B verification):**

1. **Reorder edge arrays so C2E indices are contiguous (BOTH platforms).**
   Previously dismissed for MI300A based on "85.7% cache line utilization" —
   but the per-kernel coalescing number is **27% of peak, not 85.7%**. The
   85.7% was measuring something else (cache line fill from any source). This
   is the highest-impact item: directly attacks the bottleneck on both archs.

2. **Tune register pressure on GH200** — maxnreg=80 already used in old gt4py PR.
   Verify it's still in effect; lower may trade occupancy for spilling.

3. **Re-test `(256,1,1)` carefully on GH200** — only briefly tested (~4% gain).
   Cell-consecutive threads sharing C2E lookups should help GH200 L1 hit rate.

4. ~~LDS staging~~ — last resort; #1 attacks the same problem more directly without
   manual data movement.

Observations (directly measured, no derivations):
- **HBM utilization:** GH200 at 89% of 4 TB/s peak (ncu); MI300A at 70% of
  3.47 TB/s peak (rocprof-compute §4.1.9). GH200 is closer to its HBM ceiling.
- **GH200 wins ~35% wall-clock** on this kernel (121 μs vs 187 μs).
- **Both kernels stall on memory.** MI300A: vL1D stalled on L2 data 48% of
  cycles (§16.2.0), L2-Fabric Read Latency 1440 cycles (§2.1.25). GH200: 84.3%
  L1TEX scoreboard stalls (ncu UI).

The earlier claim "MI300A is at 43% of peak, AMD caches absorb more" was based
on extract_pmc.py output which disagrees with rocprof-compute analyze by 1.62×
for unknown reasons (see HBM BW reconciliation note above). The number quoted
here (70%) is directly from rocprof-compute analyze §4.1.9.

**vL1D / per-CU L1 caching on MI300A:**

| Metric | map_100_fieldop_1 (256,1,1) |
|--------|------------------------------|
| vL1D Cache Hit Rate | **72.41%** |
| vL1D Coalescing | **27.03% of peak** ⚠️ |
| vL1D BW | 16984 Gb/s (27.71% of peak) |
| vL1D Stalled on L2 Data | 48.31% of cycles |

vL1D catches a lot of repeat reads (72% hit), but **its bandwidth efficiency
is poor (27% coalescing)** — the C2E gather scatters lanes across edge memory.
GH200's equivalent symptom: 86.5% sector utilization on missed L1 sectors and
"only 27.7 of 32 bytes utilized per sector" warning from ncu.

**Implication:** the C2E gather pattern is the shared bottleneck across both
platforms. Improving its coalescing/sector utilization is the most promising
code-side optimization for both.

**LDS conclusion (more nuanced):** earlier I claimed "vL1D absorbs everything,
LDS won't help." That overstated the case. vL1D **does** catch reuse but is
bandwidth-inefficient at it. LDS staging could in principle help **if** combined
with a deduplication scheme that reduces the number of distinct edges fetched
per cell-block. The synthetic LDS test was neutral, but it didn't deduplicate.
A `__shfl`-based or LDS-based deduplication of C2E indices is still untested
and may help — but reordering edge arrays first is the simpler attack on the
same problem.

### L2 cache analysis (baseline 32,8 — historical)

| Kernel | TCC_MISS | TCC_HIT | L2 Hit Rate |
|--------|----------|---------|-------------|
| map_100_1 | 58334 | 11402 | 16.4% |
| map_115_1 | 54419 | 10625 | 16.3% |
| map_60 | 36256 | 7122 | 16.4% |
| map_31 (no C2E) | 12466 | 1837 | 12.7% |
| map_0 (with C2E) | 17730 | 4326 | 19.8% |

These are baseline numbers. With (256,1,1), L2 hit rate jumps to ~47% on the heavy
kernels (verified — see the bandwidth table above).

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

## Key Findings

### 1. DaCe-generated kernels are mostly bandwidth-bound, with a coalescing gap
Two distinct bandwidth measurements at different points of the memory hierarchy
(numbers below are for map_100_fieldop_1 at `(256,1,1)`, from `rocprof-compute analyze`):
- **Demand BW** (kernel asks for, counted from GCN assembly × thread count): **3462 GB/s,
  94% of MI300A's 3668 GB/s peak**. Kernel issues enough loads to saturate.
- **L2-Fabric / HBM-side BW**: **2.43 TB/s, 70% of 3.47 TB/s peak**. Caches absorb
  ~37-44% of demand bytes; the rest goes to HBM.
- **vL1D Coalescing**: only **27% of peak** ⚠️. Per-CU L1 issues many requests
  per element due to scattered C2E gather — bandwidth-inefficient even though
  hit rate is 72%.

The kernel is near HBM saturation (70% of peak), so removing more demand bytes
or improving coalescing efficiency (so each vL1D request retrieves more useful
bytes) would directly translate to less HBM traffic and a faster kernel.

### 2. The MI300A vs GH200 gap shrinks substantially with `(256,1,1)`
- The original 1.4-2.1x per-kernel gap from AMD_INTRODUCTION.md was largely a block
  size mismatch — the DaCe default `(32,8)` is NVIDIA-friendly but wrong for AMD.
- After setting `gpu_block_size_2d=(256,1,1)`, the gap on the solver shrinks to **1.13x**.
- For map_100_fieldop_1 specifically (directly measured, no derivation):
  GH200 at 89% of 4 TB/s HBM peak, MI300A at 70% of 3.47 TB/s HBM peak.
  GH200 is closer to its HBM ceiling. Whether the remaining gap is fully
  explained by hardware peak + saturation differences or by something else
  is not established — would require comparing identical kernels at identical
  HBM utilization.

### 3. Inter-kernel overhead is negligible

The previously reported "86 μs (10%) inter-kernel gap" was a calculation artifact:
it compared rocprofv3 GPU kernel time (734 μs) against pytest-benchmark wall time
(820 μs), which also includes Python/GT4Py dispatch overhead. The 86 μs reflects
that host overhead, not GPU idle time between kernels.

Measured directly from rocprofv3 (`wall_clock(first→last kernel) − sum(kernel times)`):

- Kernel sum: 648 μs
- Wall clock first→last: 660 μs
- **Inter-kernel gap: 10.8 μs (1.6%)**

**Kernel fusion would save at most ~10 μs (~1%). Not worth pursuing.**

### 4. Occupancy is NOT the bottleneck
- Already at max occupancy (8 waves/SIMD) for 10 of 12 kernels
- Forcing more waves causes 2.5x slowdown due to register spilling

### 5. C2E scatter IS the suspected bottleneck — re-investigate
- Earlier analysis "85.7% cache line utilization → not a bottleneck" was based on
  a different metric (cache-line fill rate from any source, including reuse).
- Per-kernel rocprof-compute analyze on map_100_fieldop_1 shows **vL1D coalescing
  is only 27% of peak** on MI300A. ncu shows GH200 wastes 13.5% of bytes per
  sector on the 60.8% of L1-missed sectors. Both point to the same root cause:
  threads in a wave fetch 3 edges per cell from non-contiguous edge memory.
- **Edge reordering or wavefront-level edge deduplication is now the
  highest-priority untested optimization** (see "Worth investigating" above).

### 6. L2 cache behavior depends on block size
- With baseline `(32,8)`: ~15-20% hit rate (heavy 2D kernels) — streaming pattern,
  each element read once, little reuse captured.
- With `(256,1,1)`: **L2 hit rate jumps to 32-50%** on heavy 2D kernels because
  cell-consecutive threads on the same CU share cache lines. This is the main
  driver of the 21% speedup.
- 1D / scan kernels are unaffected (block size already (64,1,1)).

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
- `amd_scripts/patch_lds_c2e_v2.py` — utility to stage C2E gather through LDS (neutral)
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
