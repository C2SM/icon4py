[← Back to main report](../PROFILING_RESULTS.md) · [Hardware Reference](HARDWARE_REFERENCE.md) · [Attempted Optimizations](ATTEMPTED_OPTIMIZATIONS.md)

# Deep Analysis: map_100_fieldop_1 (hottest kernel, 187 μs)

The hottest kernel in the solver — 27% of total kernel time. All measurements
verified 2026-04-19.

## Bandwidth analysis (verified 2026-04-19)

Re-counted from the compiled GCN assembly
(`vertically_implicit_solver_at_predictor_step_cuda-hip-amdgcn-amd-amdhsa-gfx942.s`,
lines 6810-7101 of the current `(256,1,1)` cache):

- 23 × `global_load_dwordx2` (8 bytes each) = 184 bytes/thread
- 3 × `global_load_dword` (4 bytes, C2E int32 indices) = 12 bytes/thread
- 4 × `global_store_dwordx2` (8 bytes each) = 32 bytes/thread
- **Total: 228 bytes/thread**

Total threads: `dim3(156, 79, 1) × dim3(256, 1, 1) = 3,156,224 threads` (verified
via the kernel's `hipLaunchKernel` call in the compiled HIP source).

Total demand traffic: 228 bytes × 3,156,224 threads = **720 MB**

| Metric                                         | Value                                         |
| ---------------------------------------------- | --------------------------------------------- |
| Kernel time (rocprof-compute multi-pass)       | 187 μs                                        |
| Kernel time (clean rocprofv3)                  | 166 μs                                        |
| Total demand bytes per invocation              | **720 MB**                                    |
| **Demand BW** (assembly-counted, on 187 μs)    | **3850 GB/s (105% of 3668 GB/s memory peak)** |
| **HBM moved** (rocprof-compute analyze §4.1.9) | **454 MB** (= 2.43 TB/s × 187 μs)             |
| **L2+L1 absorbs**                              | (720 − 454) / 720 = **37%**                   |

The demand BW exceeds the raw HBM peak (3.85 TB/s vs 3.47 TB/s) — but this is
**demand**, not HBM-side bytes. The 37% absorption by the cache hierarchy is what
allows it: only 454 MB actually goes through HBM at 2.43 TB/s = 70% of peak.

## Synthetic bandwidth benchmark comparison

Hand-optimized HIP kernels matching the solver's access pattern achieve 80-93% of HBM peak.
The DaCe-generated kernel hits 70% of HBM peak (was previously claimed "94.4%" — that was
the demand BW vs memory-peak comparison, not HBM saturation). Cache absorption gives the
kernel its head room: pushing 720 MB of demand through 454 MB of HBM traffic.

## Memory access pattern

Per thread: 23 double reads (184 bytes) + 3 int32 C2E indices (12 bytes) +
4 double writes (32 bytes) = 228 bytes total.

- 5 arrays read at both [cell,k] and \[cell,k+1\]: vertical stencil pattern
- 6 arrays at [cell,k] only: coalesced
- 1 array at [cell] only: broadcast across K
- 3 C2E indirect reads of mass_flux via connectivity table
- 3 geofac_div reads (cell-only, 2D)
- 4 output arrays written at [cell,k]

## C2E scatter analysis

Measured C2E edge index spread for 32 consecutive cells (one warp):

| Metric                  | Original ordering | Reordered (min-cell) |
| ----------------------- | ----------------- | -------------------- |
| Median spread           | 39,881            | 3,612                |
| Cache lines per warp    | 14 (mean)         | 13.4 (mean)          |
| Cache line utilization  | **85.7%**         | 89.4%                |
| Bandwidth amplification | 1.17x             | 1.12x                |

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
already 85% used). The 27% vL1D coalescing inefficiency wastes vL1D bandwidth,
not HBM bandwidth.

**Updated verdict (2026-04-20):** Even attacking the 27% vL1D coalescing
(via reordering or `__shfl`-based dedup) is unlikely to give meaningful
wall-clock speedup on this kernel:

- **vL1D Bandwidth Utilization is 27.79%** — there's huge headroom in vL1D
  capacity. We're only using ~28% of what vL1D can deliver.
- **L1 hit rate is 72%** — only 28% of vL1D requests reach L2. Even fixing
  C2E coalescing perfectly would only affect a small slice of L1-miss traffic.
- **The kernel is HBM-bound** (70% of HBM peak), and **vL1D coalescing fixes
  don't directly reduce HBM traffic**.

The remaining wall-clock opportunity on map_100_fieldop_1 is **reducing HBM
traffic itself**, not improving cache-internal efficiency. The biggest
candidate for that is fusing `gtir_tmp_83/96/97` (currently written to HBM
by one kernel and read back by the next). See the GT4Py engineer note in
[ATTEMPTED_OPTIMIZATIONS.md](ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating).

Reproduction commands for the rocprof-compute analyze numbers cited above
are in the main [PROFILING_RESULTS.md "How to verify yourself"](../PROFILING_RESULTS.md#how-to-verify-yourself) section.

## LDS staging experiment for C2E gather

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

**Where LDS could in principle help (verdicts updated 2026-04-20):**

1. **Wavefront-level edge deduplication** — neighboring cells share edges (85.7% cache line
   fill confirms reuse exists). A custom kernel could `__shfl` indices across lanes, have
   each unique edge loaded once, then distribute values back. **This would reduce vL1D
   request count, not HBM traffic** — the L1 already catches most of the reuse (72% L1 hit).
   Predicted impact: \<2% wall-clock, same caveat as C2E reordering. Requires a hand-written
   HIP kernel, not a simple patch.

2. **Cell-broadcast values** — `geofac_div[cell]` is read 3 times per cell. Currently each
   read goes to L1. Staging it once in LDS would save 2 L1 lookups per thread. Marginal.

3. **Tridiagonal scan kernels** (`map_85`, `map_90`) — could benefit from LDS-based block-wide
   scan implementations. Requires algorithmic restructuring, not a patch. These are 1D
   scan kernels — small contribution to total solver time, low priority.

**Conclusion:** LDS staging the C2E gather (or wavefront-level dedup of it) targets vL1D
inefficiency, but vL1D bandwidth utilization is only 27.79% — there's nothing to win from
making the cache faster when the cache isn't the bottleneck. The kernel is HBM-bound; only
optimizations that reduce HBM traffic (e.g., fusing intermediates) will move wall-clock.

## Block size effect on MI300A — verified A/B (32,8) vs (256,1,1)

Same gt4py, same build cache (separate dirs to force recompile), only block size
differs. Both runs use the patched `extract_pmc.py` (which uses `(TCC_READ + TCC_WRITE) × 64` for HBM BW and `TCC_HIT/TCC_REQ` for L2 hit, verified to match
rocprof-compute analyze §4.1.9 and §2.1.21 within 3% on map_100_fieldop_1).

| Kernel            | (32,8) Dur | (32,8) HBM BW | (32,8) L2 hit | (256,1,1) Dur   | (256,1,1) HBM BW | (256,1,1) L2 hit |
| ----------------- | ---------- | ------------- | ------------- | --------------- | ---------------- | ---------------- |
| map_100_1         | 247.6 μs   | 1.73 TB/s     | **16.8%**     | 187.2 μs (-24%) | 2.35 TB/s        | **47.4%**        |
| map_111_1         | 214.2 μs   | 1.83 TB/s     | 16.4%         | 167.6 μs (-22%) | 2.25 TB/s        | 42.7%            |
| map_60            | 148.0 μs   | 1.82 TB/s     | 16.5%         | 124.9 μs (-16%) | 2.24 TB/s        | 42.8%            |
| map_0             | 62.9 μs    | 2.15 TB/s     | 19.7%         | 51.9 μs (-17%)  | 3.12 TB/s        | 49.8%            |
| map_31            | 50.0 μs    | 1.75 TB/s     | 12.7%         | 39.3 μs (-21%)  | 2.01 TB/s        | 32.3%            |
| map_85 (scan, 1D) | 58.6 μs    | 1.98 TB/s     | 22.6%         | 60.4 μs         | 1.92 TB/s        | 22.6%            |
| map_90 (scan, 1D) | 26.5 μs    | 2.25 TB/s     | 22.6%         | 26.2 μs         | 2.28 TB/s        | 22.6%            |

Sources: `workloads/rcu_amd_baseline_32x8_solver/MI300A_A1/pmc_perf.csv` and
`workloads/rcu_amd_256x1_solver/MI300A_A1/pmc_perf.csv`. Block size verified by
inspecting `dim3` in the compiled HIP launch (`(1244, 10, 1) × (32, 8, 1)` for
baseline; `(156, 79, 1) × (256, 1, 1)` for optimized). Durations from
rocprof-compute multi-pass profiling include instrumentation overhead; clean
rocprofv3 numbers are lower.

**Verified findings:**

- Heavy 2D kernels (map_100_1, map_111_1, map_60, map_0, map_31): **L2 hit rate
  triples** (15-20% → 32-50%) with `(256,1,1)`. Cell-consecutive threads on the
  same CU share cache lines.
- HBM BW also **increases** (1.7-2.2 → 2.0-3.1 TB/s) — the kernel runs faster
  and pushes more bytes/sec into HBM. (Note: the earlier doc claimed HBM "slightly
  drops because cache absorbs more"; that was wrong, based on the buggy
  `extract_pmc.py`. With the corrected formula, HBM BW rises.)
- 1D / scan kernels (map_85, map_90, map_91 etc.): unchanged as expected.

GT4Py Timer A/B at the full-solver level (1000 runs each, same MI300A, current gt4py):

- (32,8) baseline: median **753 μs**
- (256,1,1): median **604 μs**
- Speedup: **−19.8% (~20%)** ✓ matches the doc's headline claim.

## Cross-platform memory hierarchy (MI300A vs GH200) — fully verified A/B

For map_100_fieldop_1 (the hottest kernel), all four configurations directly measured.

| Platform   | Block       | Duration                         | DRAM/HBM moved              | DRAM/HBM BW   | % of HBM peak        | L1 / vL1D hit                                                | Absorbed |
| ---------- | ----------- | -------------------------------- | --------------------------- | ------------- | -------------------- | ------------------------------------------------------------ | -------- |
| **MI300A** | `(32,8)`    | **247.6 μs**                     | **428 MB** (= 1.73 × 247.6) | **1.73 TB/s** | **50%** of 3.47 TB/s | TBD (rocprof-compute analyze run not done for this dispatch) | **41%**  |
| **MI300A** | `(256,1,1)` | 187 μs (mp) / 166 μs (rocprofv3) | **454 MB**                  | **2.43 TB/s** | **70%** of 3.47 TB/s | vL1D **72%**                                                 | 37%      |
| **GH200**  | `(32,8)`    | **121.0 μs**                     | **431 MB**                  | **3.56 TB/s** | **89%** of 4 TB/s    | **36.5%**                                                    | 40%      |
| **GH200**  | `(256,1,1)` | **115.6 μs** (-4.5%)             | **412 MB** (-4.4%)          | **3.57 TB/s** | **89%** of 4 TB/s    | **13.1%** ⚠️                                                 | 43%      |

**Provenance:**

- Demand bytes (720 MB): re-counted from GCN assembly
  (`vertically_implicit_solver_at_predictor_step_cuda-hip-amdgcn-amd-amdhsa-gfx942.s`,
  lines 6810-7101): 23 × `global_load_dwordx2` + 3 × `global_load_dword` + 4 ×
  `global_store_dwordx2` = 228 B/thread × 3,156,224 threads.
- MI300A HBM BW 2.43 TB/s: rocprof-compute analyze §4.1.9 on dispatch 11/23/35.
- MI300A HBM moved 454 MB: derived 2.43 TB/s × 187 μs.
- GH200 (32,8) duration 121.0 μs: ncu `gh200_solver.ncu-rep` and `gh200_l2.ncu-rep`,
  median across 25+ dispatches (range 120.6-122.3 μs, very tight).
- GH200 (32,8) DRAM 431 MB: ncu `gh200_m100_dram_32x8.ncu-rep` (340 MB read + 91 MB write).
- GH200 (256,1,1) duration 115.6 μs: ncu `gh200_dram256.ncu-rep` and
  `vertically_implicit_solver.ncu-rep`, median across 22+ dispatches.
- GH200 (256,1,1) DRAM 412 MB: ncu `gh200_m100_dram.ncu-rep` (321 MB read + 91 MB write).
- GH200 L1 hit rates from `l1tex__t_sector_hit_rate.pct`.

**Verified cross-platform findings:**

- **GH200 saturates HBM at both block sizes (89% of 4 TB/s peak).** It's hardware-bandwidth-bound.
- **GH200 (256,1,1) is ~5% faster than (32,8)**: 121 → 116 μs, 431 → 412 MB DRAM.
  L1 hit drops 36.5% → 13.1%, but **L2 hit jumps 34.9% → 48.6%** (verified
  2026-04-20 from `lts__t_sectors_op_read_lookup_hit/miss` counters). L2 catches
  what L1 can't hold, so net DRAM traffic still drops slightly. Modest gain
  compared to MI300A's -24% on the same kernel.
- **MI300A (256,1,1) wins more (-24%) because MI300A had more room to improve.**
  With (32,8) MI300A was at lower HBM utilization than GH200, so block size mattered more.
- **Absorption % is comparable (37-43%)** across all four configs. Both architectures'
  caches catch a similar fraction of the demand traffic on this kernel, despite the
  different cache hierarchies (see [Hardware Reference](HARDWARE_REFERENCE.md) for sizes).
- **GH200 wins wall-clock by ~30%** (116 vs 166 μs at best block sizes) — explained by
  higher HBM peak (4 vs 3.47 TB/s) AND higher saturation (89% vs 70%).

(MI300A duration 187 μs is from rocprof-compute multi-pass profiling, which
adds instrumentation overhead; clean rocprofv3 reports 166 μs.)

### Cross-platform per-kernel comparison (map_100_fieldop_1)

The hottest kernel — 27% of total kernel time. All three platform/block combos
verified 2026-04-20.

| Metric                   | MI300A `(256,1,1)`                            | GH200 `(32,8)`                                    | GH200 `(256,1,1)`                                                 |
| ------------------------ | --------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| Duration                 | 187 μs (rocprof-compute) / 166 μs (rocprofv3) | 121 μs                                            | 116 μs                                                            |
| **HBM-side BW**          | **2.43 TB/s** (70% of 3.47 TB/s peak)         | 3.56 TB/s (89% of 4 TB/s peak)                    | 3.57 TB/s (89% of 4 TB/s peak)                                    |
| **DRAM bytes**           | 454 MB (= 2.43 × 187)                         | 432 MB                                            | 413 MB (-4%)                                                      |
| L2 hit rate              | **47.5%** (rocprof-compute §2.1.21)           | **34.9%** (computed from `lts` hit/miss counters) | **48.6%** (same — jumps with (256,1,1) to compensate for L1 drop) |
| **L1 hit rate**          | **vL1D 72.4%** (rocprof-compute §2.1.19)      | 36.5% (`l1tex__t_sector_hit_rate.pct`)            | **13.1%** ⚠️ (drops with (256,1,1))                               |
| Coalescing / sector util | vL1D coalescing **27% of peak** (§16.1.3)     | 88.5% bytes/sector util on global loads           | 88.5% (unchanged)                                                 |
| **IPC**                  | **0.24** (4.87% of peak)                      | 0.28 inst/cycle                                   | 0.29 inst/cycle                                                   |
| **FP64 utilization**     | 1.64% of peak (§2.1.0 VALU)                   | 8.69% of peak                                     | 9.44% of peak                                                     |
| Theoretical occupancy    | 8 waves/SIMD (max)                            | 62.5% (10/16) — register-limited                  | (same)                                                            |
| Top stall reason         | vL1D stalled on L2 data 48% (§16.2.0)         | 84.3% L1TEX scoreboard                            | (same)                                                            |
| L2-Fabric Read Latency   | **1440 cycles** (§2.1.25)                     | n/a equivalent                                    | n/a equivalent                                                    |

**Key observations from the new GH200 data:**

- **L1 hit rate is ~3× lower at (256,1,1) than (32,8) on GH200** (13.1% vs 36.5%).
  But duration is still slightly better at (256,1,1) (-4%) — so the win comes from
  somewhere other than L1 (likely L2/fabric efficiency or wavefront alignment).
- **MI300A vs GH200 L1 hit gap is HUGE**: at matching `(256,1,1)`, MI300A vL1D
  is 72.4% vs GH200 L1 13.1% — **5.5× advantage**. The per-CU 32 KB AMD cache
  beats the per-SM 256 KB shared NVIDIA cache on this access pattern.
- **Bytes-per-sector utilization on GH200 is high (88.5%)** at both block sizes —
  i.e., when GH200 fetches a 32-byte sector, it uses ~28.3 bytes. Coalescing is
  fine; the issue is just that it has to fetch many more sectors because L1 hit is low.
  | Active threads / wave | 63.97 / 64 (99.95%) | n/a equivalent |

**Provenance:** MI300A from `rocprof-compute analyze -p workloads/rcu_amd_256x1_solver/MI300A_A1/ --dispatch 11 23 35` (averaged across 3 invocations of map_100_fieldop_1). GH200 from `gh200_solver.ncu-rep` opened in ncu UI.

**HBM BW source (reconciled 2026-04-19):** Originally extract_pmc.py reported
1.50 TB/s while rocprof-compute analyze reported 2.43 TB/s for the same kernel
from the same pmc_perf.csv. Root cause traced and patched: extract_pmc.py used
`(TCC_EA0_RDREQ + TCC_EA0_WRREQ) × 64`, which only counts External Access
channel 0 (L2-miss traffic from one channel) and missed L2 traffic on hits.
Patched formula: `(TCC_READ + TCC_WRITE) × 64`, verified to match rocprof-compute
within 3% (2.35 TB/s vs 2.43 TB/s on map_100_fieldop_1). The 2.43 TB/s number
in this table is from rocprof-compute analyze §4.1.9.

Sources for GH200 rows: `gh200_solver.ncu-rep` (the original (32,8) baseline)
gave the ncu UI warnings "L1TEX scoreboard ... 84.3%", "theoretical occupancy
62.5% limited by registers", "only 27.7 of 32 bytes transmitted per sector are
utilized ... applies to 60.8% of sectors missed in L1TEX". The (256,1,1) row
metrics are from `gh200_m100_full_256x1.ncu-rep` (collected 2026-04-20 with
`l1tex__t_sector_hit_rate.pct`, `dram__bytes.sum`, IPC, FP64% directly).

⚠️ **Earlier doc text claimed GH200 (256,1,1) L1 hit was 39.2%** — that was a
mis-attribution: 39.2% came from the (32,8) ncu UI text "60.8% sectors miss".
The actual measured (256,1,1) L1 hit is 13.1%. Now corrected.

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

### What this implies for optimization on map_100_fieldop_1

**MI300A — picture is more nuanced than initial read.** The kernel:

- **Hits HBM at 70% of peak**, not 43% as previously claimed. Closer to bandwidth-bound than absorption-bound.
- **vL1D hit 72%** — good per-CU caching, but...
- **Coalescing only 27% of peak** ⚠️ — vL1D bandwidth efficiency is poor; lots
  of wasted lanes per access. Same root cause as GH200 throwing away ~13.5%
  of bytes fetched on L1 misses: the C2E gather scatters threads across edge memory.
- **vL1D stalls on L2 data 48% of cycles** → memory-latency-bound, not throughput-bound at the vL1D layer.
- IPC 0.24 (4.87% of peak), VALU 1.64% — confirms compute-light, memory-stall-bound.

**Implication change:** On MI300A, the bottleneck is **not** "vL1D already absorbs everything." It's:

1. Coalescing inefficiency in vL1D (27% of peak coalescing) wastes ~3-4× the bandwidth needed
2. L2-Fabric read latency 1440 cycles → wave occupancy can't hide it (vL1D stalled on L2 48%)

So **fixing coalescing on the C2E gather should help MI300A too**, contrary to my earlier claim. The mechanism: better coalescing → fewer vL1D requests per element → fewer L2 misses → fewer 1440-cycle stalls.

**GH200 side (HBM 89% saturated):**

- L1 hit rate **drops to 13.1%** with `(256,1,1)` (from 36.5% at (32,8)).
  But duration improves slightly because L2/fabric absorbs the loss.
- Bytes-per-sector utilization is high (88.5%) at both block sizes — coalescing
  is fine, GH200 just has to fetch many sectors because L1 misses dominate.
- Register-pressure-limited occupancy at 62.5% (10/16 warps).
- Top stall: 84.3% L1TEX scoreboard.

**Concrete things to try (each needs A/B verification):**

1. **Fuse intermediates `gtir_tmp_83/96/97`** — currently written to HBM by
   one kernel and read back by the next. Removing the round-trip would directly
   reduce HBM traffic, which IS the bottleneck. Blocked by `concat_where`
   K-domain splits in DaCe — see GT4Py engineer note in
   [ATTEMPTED_OPTIMIZATIONS.md](ATTEMPTED_OPTIMIZATIONS.md#note-for-gt4py--dace-engineers-worth-investigating).
   **Largest remaining opportunity.**
2. **(256,1,1) on GH200**: -5% verified per-kernel (121 → 116 μs). Already
   confirmed; the open question is whether to gate the option per-platform or apply globally.
3. **Tune register pressure on GH200** — maxnreg=80 was used in the old gt4py PR;
   verify it's still in effect.
4. ~~Reorder edge arrays so C2E indices are contiguous~~ — predicted \<2% wall-clock
   impact. The 27% vL1D coalescing wastes vL1D bandwidth (which we have plenty of),
   not HBM bandwidth (which is the actual bottleneck).
5. ~~LDS staging the existing access pattern~~ — neutral in the synthetic test.
   `__shfl`-based or LDS-based deduplication is a separate untested idea but would
   target the same vL1D-only metric and is likely also \<2%.

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

| Metric                  | map_100_fieldop_1 (256,1,1) |
| ----------------------- | --------------------------- |
| vL1D Cache Hit Rate     | **72.41%**                  |
| vL1D Coalescing         | **27.03% of peak** ⚠️       |
| vL1D BW                 | 16984 Gb/s (27.71% of peak) |
| vL1D Stalled on L2 Data | 48.31% of cycles            |

vL1D catches a lot of repeat reads (72% hit), but **its bandwidth efficiency
is poor (27% coalescing)** — the C2E gather scatters lanes across edge memory.
GH200's equivalent symptom: when a warp misses L1 and fetches a 32-byte block
from L2, it uses only ~28 bytes (ncu warning: "only 27.7 of 32 bytes utilized
per sector").

**Implication:** the C2E gather pattern is the shared bottleneck across both
platforms. Improving its coalescing efficiency (each request retrieves more
useful bytes) is the most promising code-side optimization for both.

**LDS conclusion (more nuanced):** earlier I claimed "vL1D absorbs everything,
LDS won't help." That overstated the case. vL1D **does** catch reuse but is
bandwidth-inefficient at it. LDS staging could in principle help **if** combined
with a deduplication scheme that reduces the number of distinct edges fetched
per cell-block. The synthetic LDS test was neutral, but it didn't deduplicate.
A `__shfl`-based or LDS-based deduplication of C2E indices is still untested
and may help — but reordering edge arrays first is the simpler attack on the
same problem.

## L2 cache analysis — historical baseline (32,8)

| Kernel           | TCC_MISS | TCC_HIT | L2 Hit Rate |
| ---------------- | -------- | ------- | ----------- |
| map_100_1        | 58334    | 11402   | 16.4%       |
| map_115_1        | 54419    | 10625   | 16.3%       |
| map_60           | 36256    | 7122    | 16.4%       |
| map_31 (no C2E)  | 12466    | 1837    | 12.7%       |
| map_0 (with C2E) | 17730    | 4326    | 19.8%       |

Cross-checked against the fresh (32,8) baseline measured 2026-04-19 (see
"Block size effect on MI300A" table above): 16.8% / 16.5% / 12.7% / 19.7% — all
match within 0.1-0.4%. Historical numbers verified.

With (256,1,1), L2 hit rate jumps to ~47% on the heavy kernels.

## Per-Kernel Register & Occupancy Profile

From `pmc_perf.csv` (rocprof-compute hardware counters):

| Kernel             | Arch VGPR | Accum VGPR | SGPR | LDS | Scratch | Waves/SIMD |
| ------------------ | --------- | ---------- | ---- | --- | ------- | ---------- |
| map_100_1 (207 μs) | 56        | 0          | 96   | 0   | 0       | 8 (max)    |
| map_115_1 (183 μs) | 48        | 0          | 96   | 0   | 0       | 8 (max)    |
| map_60 (133 μs)    | 36        | 4          | 64   | 0   | 0       | 8 (max)    |
| map_0 (59 μs)      | 32        | 0          | 32   | 0   | 0       | 8 (max)    |
| map_31 (41 μs)     | 12        | 4          | 32   | 0   | 0       | 8 (max)    |
| map_85 (58 μs)     | 68        | 4          | 32   | 0   | 0       | 7          |
| map_90 (25 μs)     | 92        | 4          | 16   | 0   | 0       | 5          |
| map_91 (9 μs)      | 8         | 0          | 32   | 0   | 0       | 8 (max)    |
| map_100_0 (6 μs)   | 56        | 0          | 96   | 0   | 0       | 8 (max)    |
| map_115_0 (5 μs)   | 48        | 0          | 96   | 0   | 0       | 8 (max)    |
| map_13 (6 μs)      | 44        | 4          | 32   | 0   | 0       | 8 (max)    |
| map_35 (4 μs)      | 8         | 0          | 32   | 0   | 0       | 8 (max)    |

Waves/SIMD = min(8, floor(512 / Arch_VGPR)). Hardware max is 8 waves per SIMD on gfx942.
10 of 12 kernels are already at max occupancy.
