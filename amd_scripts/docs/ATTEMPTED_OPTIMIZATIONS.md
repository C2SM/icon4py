[← Back to main report](../PROFILING_RESULTS.md) · [Hardware Reference](HARDWARE_REFERENCE.md) · [Deep Analysis](DEEP_ANALYSIS.md)

# Attempted Optimizations — what was tried, what worked, what didn't

Detail of every optimization explored on the solver kernel. Headline summary
table is in the [main report](../PROFILING_RESULTS.md). This file has the full
A/B sweep tables, methodology, and per-experiment notes.

## GPU Block Size Tuning

Date: 2026-04-14

The headline result is in the [main report](../PROFILING_RESULTS.md). This
section drills into which specific `gpu_block_size_*d` setting is responsible.

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

## DaCe Fusion Analysis

### What was tried
- `MapFusion`: applied 9 times (inner tlet maps), but cannot fuse the main maps because
  intermediates (`gtir_tmp_83`, `_96`, `_97`) have multiple consumers
- `SubgraphFusion`: cannot apply — "nodes between maps with incoming edges from outside"
- Root cause: `concat_where` + `broadcast` in GT4Py field operators creates K-domain splits
  (`_0` for K=0, `_1` for K=1..79), producing multiple producers per intermediate

### Note for GT4Py / DaCe engineers (worth investigating)

The "inter-kernel fusion saves 1%" verdict (see Key Findings in the main report)
is about **launch overhead** — verified small. A separate motivation that has
**NOT** been measured but is plausible:

The unfused intermediates (`gtir_tmp_83`, `_96`, `_97`) are **written to HBM by one
kernel and read back by the next**. On this kernel HBM is the bottleneck (70% of
peak), so any reduction in HBM traffic translates directly to wall-clock savings.
If `MapFusion`/`SubgraphFusion` could be unblocked — by either:
- restructuring `concat_where` so it doesn't produce multiple-producer intermediates, or
- relaxing the "incoming edges from outside" check in `SubgraphFusion` for the
  K-domain-split pattern,

then the fused region could keep these intermediates in registers/scratch instead of
round-tripping through HBM. The expected savings depend on the size of those
intermediates relative to the 720 MB demand bytes per invocation, which we
haven't measured. Worth flagging because:
1. The structural blocker (`concat_where` K-splits) is the same across many
   GT4Py stencils, not just this one.
2. The HBM-bandwidth-bound nature of the kernel makes any demand-byte reduction
   directly visible in wall-clock time.

This is a **GT4Py/DaCe pipeline concern**, not a tuning knob. Not in scope for
the current solver tuning effort, but flagged here so the gt4py team has the
context if they revisit fusion.

### `fuse_tasklets` optimization
- Added `fuse_tasklets=True` to `model_options.py` for the solver stencil
- Fuses tasklet operations within existing map scopes

**Status: neutral on current gt4py.** Historical context:

| gt4py version | A/B improvement | Notes |
|---|---|---|
| 1.1.4 (`amd_profiling` branch, 30 runs) | ~7% (0.797 → 0.732 ms) | When first tested |
| Newer gt4py (Edoardo's measurement) | ~1.5% (0.782 → 0.770 ms) | Newer optimization pipeline absorbed most of the benefit |
| Current gt4py (this study, 1000 runs) | **neutral** (within noise) | See "What does NOT help" table in main report |

The newer DaCe pipeline already fuses tasklets that previously needed
`fuse_tasklets=True` to be explicitly requested.

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

## GH200 vs MI300A Synthetic Bandwidth Comparison

Same synthetic kernels run on both platforms:
- **Simple stream**: MI300A wins (0.016 vs 0.017 ms) — competitive raw HBM BW
- **Multi-array patterns**: GH200 is consistently **1.23-1.33x** faster
- The gap is hardware-level: GH200's memory controller handles many concurrent streams
  more efficiently

Block size sweep on MI300A synthetic benchmark confirmed that `256×1` is **19% faster**
than the DaCe default `32×8`, which motivated the `gpu_block_size=(256,1,1)` change
in icon4py.

See appendix below for detailed benchmark tables.

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
