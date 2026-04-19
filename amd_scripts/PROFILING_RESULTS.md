# MI300A Profiling Results — vertically_implicit_solver_at_predictor_step

Date: 2026-03-30 (initial), 2026-04-19 (latest update)

> **This is the executive summary.** Detailed evidence lives in:
> - [docs/HARDWARE_REFERENCE.md](docs/HARDWARE_REFERENCE.md) — vendor specs + on-node sysinfo for MI300A and GH200
> - [docs/DEEP_ANALYSIS.md](docs/DEEP_ANALYSIS.md) — per-kernel deep dive on map_100_fieldop_1, cross-platform A/B tables, demand-byte derivation
> - [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md) — every optimization tried (block size, fusion, occupancy, loop blocking, LDS, CSE) with full A/B sweeps

## Summary

Profiled the `vertically_implicit_solver_at_predictor_step` GT4Py program on
MI300A (Beverin, gfx942, ROCm 7.1.0) and GH200 (Santis, sm_90).

### Headline result

A one-line config change — `gpu_block_size_2d=(256,1,1)` for ROCm in `model_options.py`
— gives **20% solver speedup** on MI300A and closes most of the MI300A vs GH200 gap.

| Platform | Config | GT4Py Timer median (1000 runs) |
|----------|--------|-------------------------------|
| MI300A | DaCe default `(32,8,1)` | 0.753 ms |
| MI300A | **`gpu_block_size_2d=(256,1,1)`** | **0.604 ms (-19.8%)** |
| GH200 | defaults `(32,8,1)` | 0.559 ms |
| GH200 | `gpu_block_size_2d=(256,1,1)` | per-kernel: -5% on map_100_fieldop_1; full-solver GT4Py Timer not yet measured |

Gap MI300A → GH200: **1.35x → 1.08x** (at MI300A's best vs GH200's default).
Block-size tuning helps MI300A much more than GH200 — see
[Cross-platform table](docs/DEEP_ANALYSIS.md#cross-platform-memory-hierarchy-mi300a-vs-gh200--fully-verified-ab)
for per-kernel detail.

### Cross-platform story (in plain language)

**The big picture (all numbers verified):**

1. **GH200 wins by ~30% per kernel today** (116 μs vs 166 μs on map_100_fieldop_1
   at each platform's best block size).
2. **The win is mostly hardware**: GH200 has a higher HBM peak (4.0 vs 3.47 TB/s)
   AND saturates more of it (89% vs 70% on this kernel).
3. **Block size matters more on AMD**: switching (32,8)→(256,1,1) gives MI300A
   −24% on this kernel and ~−20% on the full solver. Same change on GH200 only
   gives −5%, because GH200 was already nearly bandwidth-saturated.
4. **Both architectures are bandwidth-bound, not compute-bound**: IPC ~0.2-0.3,
   VALU usage ~1-2%. The kernel waits on memory, not on math.
5. **Both architectures' caches absorb a similar fraction (37-43%)** of demand bytes.
   AMD's per-CU vL1D catches more reuse than GH200's L1 (72% vs 39% hit rate),
   but GH200 wins overall because its HBM is faster anyway.

### Can MI300A catch GH200?

**The hardware-bound part of the gap is real and won't disappear.** GH200 is
already at 89% of its 4.0 TB/s HBM peak; MI300A is at 70% of its 3.47 TB/s peak.
Even if MI300A reached 100% of its HBM peak, the HBM-peak ratio alone (3.47 vs
4.0 = 0.87) would leave a residual gap.

**Where MI300A still has headroom:**
- **HBM utilization at 70% vs GH200's 89%** — room to push more data per second.
- **vL1D coalescing only 27% of peak** — improving it means fewer vL1D requests
  per element → fewer L2 misses → less HBM traffic → faster. Same root cause
  ncu flags on GH200.

**Hardware floor (HBM-peak ratio):** MI300A 3.47 TB/s vs GH200 4.0 TB/s → **0.87**.
Even at perfect saturation, MI300A would still be ~13% behind from hardware
alone on bandwidth-bound kernels. Beating GH200 across the full model would
require per-kernel tuning (block size, registers, occupancy) since each kernel
has its own bottleneck profile.

**The biggest unfinished optimization** is C2E edge reordering or wavefront-level
edge deduplication. Same root cause hurts both architectures (MI300A: 27% vL1D
coalescing; GH200: 13.5% wasted bytes per L1-missed sector). Untested.

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

**Important caveat:** `(256,1,1)` aligns access to wavefronts, but **vL1D coalescing
on the heavy kernel is still only 27% of peak** (per-kernel rocprof-compute analyze).
The C2E gather scatters threads across edge memory even with cell-consecutive
blocks — see [Deep Analysis](docs/DEEP_ANALYSIS.md#c2e-scatter-analysis).

## Optimization Opportunities

### Confirmed helpful
1. **`gpu_block_size_2d=(256,1,1)` for ROCm** — −20% solver speedup on MI300A,
   −5% per-kernel on GH200 (see [block size tuning detail](docs/ATTEMPTED_OPTIMIZATIONS.md#gpu-block-size-tuning))

### Worth investigating
2. **Edge reordering / C2E gather coalescing** — promoted from "not helpful"
   after per-kernel rocprof-compute analyze showed vL1D coalescing is only
   **27% of peak** on map_100_fieldop_1 (MI300A) and 86.5% sector util on GH200's
   missed sectors. Highest-impact untested optimization.
3. **Grid size divisible by 6** — preliminary single-data-point shows ~4%
   additional improvement (0.617→0.594 ms on a different gt4py branch). Possibly
   related to even work distribution across MI300A's 6 XCDs. Needs A/B verification.
4. **DaCe `MapFusion`/`SubgraphFusion` for HBM-traffic reduction** (not launch
   overhead) — see GT4Py engineer notes in
   [docs/ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating](docs/ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating).

### Not helpful (verified, with measurements)
| Optimization | Result |
|--------------|--------|
| Loop blocking (K-blocking) | 2.8% on its own, redundant once `(256,1,1)` is applied |
| `fuse_tasklets` | Neutral on current gt4py (was 7% on older 1.1.4) |
| Kernel fusion (for launch overhead) | Inter-kernel gap is only 0.5-1.1% — not worth pursuing |
| Forcing higher occupancy | 2.5x slowdown (register spilling); 10/12 kernels already at max |
| Register limiting via compiler flags | Not exposed in ROCm 7.1.0 |
| CSE (common subexpression elimination) | Compiler handles it at -O3 |
| Block size `(64,6,1)` | 7% worse than `(256,1,1)` |
| LDS staging the C2E gather (synthetic) | Neutral; `__shfl`-based dedup is a separate untested idea |
| Many-array BW limitation | Synthetic benchmark proves MI300A handles 20+ arrays at 93% peak |

Full A/B sweeps and methodology in [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md).

## Per-Kernel Timing (rocprofv3 kernel-trace, median μs, MI300A)

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
| **Wall clock (first→last)** | 774 | 660 | -15% | inter-kernel gap: 0.5-1.1% |
| **GT4Py Timer median** | 760 | 604 | -21% | full SDFG call (1000 runs) |

The 2D heavy kernels (map_100_1, map_111_1, map_60, map_31) all gained 16-22% from
the (256,1,1) block size. 1D kernels and scans are unchanged as expected.

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
- After setting `gpu_block_size_2d=(256,1,1)`, the gap on the solver shrinks to **1.13x**.
- For map_100_fieldop_1 specifically: GH200 at 89% of 4 TB/s HBM peak, MI300A at 70%
  of 3.47 TB/s HBM peak. GH200 is closer to its HBM ceiling.

### 3. Inter-kernel overhead is negligible (verified A/B)
| Block size | Wall first→last | Σ kernel times | Gap | Gap % |
|---|---|---|---|---|
| (32,8) baseline | 1316 μs | 1314 μs | **6.3 μs** | **0.47%** |
| (256,1,1) | 847 μs | 840 μs | **9.1 μs** | **1.08%** |

The previously reported "86 μs (10%) inter-kernel gap" was a calculation artifact
(rocprofv3 GPU kernel time vs pytest-benchmark wall time, mixing GPU and host overhead).

**Kernel fusion would save at most ~10 μs (~1% of solver time). Not worth pursuing.**

> **TODO (cross-platform):** A first GH200 nsys trace produced an unexpectedly
> large gap (~19 μs / ~4%). Before that's reported, needs cross-checking under
> matched conditions (same iteration count; same instrumentation overhead profile).
> nsys is heavier than rocprofv3 and may itself inject inter-kernel time. Held
> back from the doc pending verification.

### 4. Occupancy is NOT the bottleneck
- Already at max occupancy (8 waves/SIMD) for 10 of 12 kernels
- Forcing more waves causes 2.5x slowdown due to register spilling

### 5. C2E scatter IS the suspected bottleneck — re-investigate
- Earlier analysis "85.7% cache line utilization → not a bottleneck" measured
  cache-line **fill** from any source, not **coalescing efficiency**.
- Per-kernel rocprof-compute analyze on map_100_fieldop_1 shows **vL1D coalescing
  is only 27% of peak** on MI300A. ncu shows GH200 wastes 13.5% of bytes per
  sector on the 60.8% of L1-missed sectors.
- **Edge reordering or wavefront-level edge deduplication is the highest-priority
  untested optimization**. See full analysis in [Deep Analysis: C2E scatter](docs/DEEP_ANALYSIS.md#c2e-scatter-analysis).

### 6. L2 cache behavior depends on block size
- With baseline `(32,8)`: ~15-20% hit rate (heavy 2D kernels) — streaming pattern.
- With `(256,1,1)`: **L2 hit rate jumps to 32-50%** on heavy 2D kernels because
  cell-consecutive threads on the same CU share cache lines. Main driver of the 20% speedup.
- 1D / scan kernels are unaffected.

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

**`extract_pmc.py` HBM BW formula** was patched 2026-04-19. Old formula
`(TCC_EA0_RDREQ + TCC_EA0_WRREQ) × 64` only counted External Access channel 0
(L2-miss traffic from one channel) and missed L2 traffic on hits — discrepancy with
rocprof-compute analyze was 1.62×. Patched formula: `(TCC_READ + TCC_WRITE) × 64`,
verified to match rocprof-compute within 3%.

## Files

- `amd_scripts/benchmark_solver.sh` — original benchmark + trace + rocprof-compute
- `amd_scripts/benchmark_solver_roofline.sh` — per-kernel roofline generation (fixed filename issue)
- `amd_scripts/benchmark_solver_occupancy.sh` — occupancy sweep via source-level attribute patching
- `amd_scripts/bw_benchmark.hip` — synthetic HIP bandwidth benchmark matching kernel patterns
- `amd_scripts/analyze_c2e_reorder.py` — C2E scatter analysis and edge reordering test
- `amd_scripts/set_waves_per_eu.py` — patch DaCe-generated HIP with waves_per_eu attribute
- `amd_scripts/patch_cse.py` — patch redundant reads (confirmed unnecessary)
- `amd_scripts/patch_lds_c2e_v2.py` — stage C2E gather through LDS (neutral)
- `amd_scripts/extract_pmc.py` — extract per-kernel duration, HBM BW, L2 hit rate from rocprof-compute pmc_perf.csv
- `workloads/rcu_amd_*/` — per-kernel profiling data (rocprof-compute output)
- `pmc_perf.csv` — hardware counter data for all kernels

---

**For full evidence and detailed measurements:**
- [docs/HARDWARE_REFERENCE.md](docs/HARDWARE_REFERENCE.md)
- [docs/DEEP_ANALYSIS.md](docs/DEEP_ANALYSIS.md)
- [docs/ATTEMPTED_OPTIMIZATIONS.md](docs/ATTEMPTED_OPTIMIZATIONS.md)
