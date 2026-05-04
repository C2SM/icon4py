# MI300A node variance — aac6 vs beverin

> ⚠️ **Provisional snapshot — 2026-05-04.** AMD confirmed they are rebuilding
> parts of the aac6 software stack, which may affect these numbers. After the
> rebuild lands, re-run the verify commands below to check whether the data
> still matches. The conclusion (chip-to-chip variance is the dominant cause)
> may need revision if the rebuild changes how aac6 nodes behave.

## Summary

Same code, same settings, on two clusters. The dominant kernel reaches very
different memory bandwidth on different chips. The fastest aac6 chip
(`ppac-pl1-s24-26`) reaches **39% of HBM peak**. A typical chip on either
cluster reaches **~30%**. The difference is mostly the chip itself, not the
cluster — same firmware on aac6, 26% spread between two chips.

## What we measured

- Solver: `vertically_implicit_solver_at_predictor_step` (icon4py commit
  `6d1866e4`, branch `amd_profiling`).
- Same gt4py + ROCm 7.2.0 + `--exclusive` on both clusters.
- Pinned each run to a specific node so we don't average across chips.

## Headline numbers

3 aac6 nodes (2–3 timer runs + 1 rocprof-compute run each); 16 beverin nodes
in the timer sweep (2 runs each), 2 of those with rocprof-compute. Solver μs
column is the GT4Py timer median. HBM number is for `map_100_fieldop_1_0_0` —
the kernel that uses 66% of the total kernel runtime. Steady power is the
p50 socket power from a 100 ms-cadence trace under load.

| Cluster / Node                    | Solver μs | HBM (TB/s) | % of HBM peak† | Steady power |
| --------------------------------- | --------- | ---------- | -------------- | ------------ |
| **aac6 ppac-pl1-s24-26**          | **514**   | **3.35**   | **39%**        | 157 W        |
| aac6 ppac-pl1-s24-16              | 565       | 2.66       | 31%            | 159 W        |
| aac6 ppac-pl1-s24-30              | 611       | 2.52       | 29%            | 184 W        |
| beverin nid002510 (fastest of 16) | 591       | 2.60       | 30%            | 174 W        |
| beverin nid002420 (slowest of 16) | 638       | 2.29       | 27%            | 181 W        |

† MI300A HBM peak is **8601.6 Gb/s** (from `rocprof-compute analyze`,
section 17.1.5). Same on both clusters.

Across 16 beverin nodes the spread is 11% (571–635 μs). Across 3 aac6 nodes
the spread is 19% (514–611 μs). The wider aac6 spread is because `-26` is
faster than anything else; the other two aac6 nodes look like typical
beverin nodes.

### Verify the numbers yourself

```bash
# on aac6
ROCP=/shared/apps/ubuntu/opt/rocm-7.2.0/bin/rocprof-compute
$ROCP analyze -p rocprof-compute_node26 -k 1 \
    | grep -E "L2.Fabric.*BW|HBM Bandwidth|HBM Read Traffic"
```

```bash
# on beverin (uenv active)
rocprof-compute analyze -p rocprof-compute_nid002510 -k 1 \
    | grep -E "L2.Fabric.*BW|HBM Bandwidth|HBM Read Traffic"
```

Read the row labeled `4.1.9 HBM Bandwidth` in the output. Compare against the
table above. The number is in Gb/s.

## What's causing the variance

1. **Different chips behave differently.** Same firmware, same driver, same
   partition mode, same workload — chips still vary by 13–26% in how fast they
   sustain the same memory traffic. We see this within both clusters: aac6 -26
   vs -16 (identical firmware) differ by 26% in HBM bandwidth; beverin's
   fastest vs slowest of 16 differ by 13%.
2. **It's not a caching difference.** The L2 hit rate is the same on every
   node for any given kernel (~47% for `map_100`). The slower chips push the
   same number of bytes through L2 and HBM — they just take longer to do it.
3. **Faster chips draw less power.** The slowest node (-30) draws 184 W
   running the same kernels that the fastest node (-26) does at 157 W. Higher
   leakage current → less work per unit of power.

### Per-kernel within-cluster comparison (silicon binning at the kernel level)

Same software, same firmware, same partition mode — just different chips.

#### Within aac6 (-26 fastest vs -30 slowest, identical firmware 04.85.90.00):

| Kernel                 | -26 dur (μs) | -30 dur (μs) | Δ time   | -26 HBM (TB/s) | -30 HBM (TB/s) |
| ---------------------- | ------------ | ------------ | -------- | -------------- | -------------- |
| map_100_fieldop_1_0_0  | 136.5        | 181.4        | **+33%** | 3.33           | 2.52           |
| map_111_fieldop_1_0_0  | 124.4        | 160.9        | +29%     | 3.43           | 2.65           |
| map_60_fieldop_0_0     | 90.2         | 118.4        | +31%     | 3.49           | 2.66           |
| map_85_fieldop_0_0     | 65.9         | 68.1         | +3%      | 2.36           | 2.29           |
| map_0_fieldop_0_0      | 43.7         | 54.7         | +25%     | 3.69           | 2.96           |
| map_31_fieldop_0_0_0_0 | 28.2         | 32.3         | +15%     | 3.75           | 3.27           |
| map_90_fieldop_0_0     | 25.1         | 25.1         | 0%       | 3.14           | 3.14           |

#### Within beverin (-510 fastest of 16 vs -420 slowest of 16, identical firmware 04.85.112.00):

| Kernel                 | -510 dur (μs) | -420 dur (μs) | Δ time | -510 HBM (TB/s) | -420 HBM (TB/s) |
| ---------------------- | ------------- | ------------- | ------ | --------------- | --------------- |
| map_100_fieldop_1_0_0  | 176.3         | 198.6         | +13%   | 2.59            | 2.29            |
| map_111_fieldop_1_0_0  | 154.9         | 171.6         | +11%   | 2.75            | 2.48            |
| map_60_fieldop_0_0     | 122.6         | 140.3         | +14%   | 2.57            | 2.24            |
| map_85_fieldop_0_0     | 54.3          | 61.8          | +14%   | 3.12            | 2.75            |
| map_0_fieldop_0_0      | 53.4          | 60.7          | +14%   | 3.02            | 2.66            |
| map_31_fieldop_0_0_0_0 | 33.5          | 36.9          | +10%   | 3.16            | 2.86            |
| map_90_fieldop_0_0     | 23.9          | 25.5          | +7%    | 3.60            | 3.38            |

L2 hit rate is identical between the two nodes for every kernel — the
bandwidth difference is pure chip variance, not caching.

The within-aac6 spread on dominant kernels (~30%) is much larger than the
within-beverin spread (~13%), driven entirely by -26 being unusually fast.
Variance also scales with kernel size: kernels >50 μs show the spread,
sub-10 μs kernels are within noise.

## What we ruled out

| Cause we suspected                                        | Result              | How we know                                                                                                                                                                                                                                        |
| --------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Compute partition (SPX vs CPX)                            | **Same**            | Both clusters: `compute_partition=SPX` in sysinfo                                                                                                                                                                                                  |
| Memory partition (NPS1 vs NPS4)                           | **Same**            | Both clusters: `memory_partition=NPS1`                                                                                                                                                                                                             |
| Maximum clock setting                                     | **Same**            | All nodes: `max_mclk=2100, max_sclk=2100`                                                                                                                                                                                                          |
| Memory clock dropping under load                          | **No**              | mclk stays at 1300 MHz the whole time, on every node we traced                                                                                                                                                                                     |
| Code or compiler differences per node                     | **Same**            | Same VGPR count, block size, kernels per cluster                                                                                                                                                                                                   |
| **Firmware / driver difference between aac6 and beverin** | **Real, but small** | aac6 SMC firmware 04.85.90.00 vs beverin 04.85.112.00; aac6 driver 6.16.6 vs beverin 6.10.5. This may shift things slightly but does **not** explain the within-aac6 difference (`-26` and `-16` have identical firmware and still differ by 26%). |

### Verify firmware and driver yourself

```bash
# on aac6
sbatch --partition=1CN192C4G1H_MI300A_Ubuntu22 --nodelist=ppac-pl1-s24-26 \
    --gpus=1 --ntasks=1 --time=00:02:00 --exclusive \
    --output=aac6_fw.out --wrap='
uname -r
cat /sys/module/amdgpu/version 2>/dev/null
/shared/apps/ubuntu/opt/rocm-7.2.0/bin/rocm-smi --showfwinfo 2>&1 | grep "GPU\[0\]"
'
cat aac6_fw.out
```

```bash
# on beverin (uenv active)
sbatch --partition=mi300 --nodelist=nid002510 \
    --uenv=b2550889de318ab5 --view=default \
    --gpus=1 --ntasks=1 --time=00:02:00 --exclusive \
    --output=beverin_fw.out --wrap='
uname -r
cat /sys/module/amdgpu/version 2>/dev/null
rocm-smi --showfwinfo 2>&1 | grep "GPU\[0\]"
'
cat beverin_fw.out
```

The most useful line is `SMC firmware version`. SMC is the small chip on the
GPU that controls clocks and power. Newer SMC firmware tends to manage power
more conservatively.

What each firmware does:

- **SMC / PM** — power management: decides clocks and voltages, throttles under load
- **MEC1 / MEC2** — command processor: receives kernel dispatches, schedules them on the CUs
- **RLC** — runlist controller: manages context switches between concurrent workloads
- **SDMA0 / SDMA1** — data movers: handle host↔device and large in-device memory transfers
- **PSP_SOSDRV / SOS** — secure boot processor: verifies firmware. Irrelevant for steady-state perf.
- **TA_RAS / TA_XGMI** — small microcontrollers for error correction (RAS) and inter-GPU coherency (XGMI). Mostly correctness, not perf.
- **VCN** — video encode/decode hardware. Irrelevant for HPC compute.

## How to run a fresh benchmark

```bash
# on aac6 — pin to a node, use a fresh cache directory
GT4PY_BUILD_CACHE_DIR=amd_aac6_repro_$(hostname -s) \
    sbatch --nodelist=ppac-pl1-s24-26 \
    --output=gt4py_repro.out --error=gt4py_repro.err \
    amd_scripts/sbatch_gt4py_timer_aac6.sh

# wait, then read the timer report
grep -A 4 "GT4Py Timer Report" gt4py_repro.out | tail -2
```

```bash
# on beverin — pin to a node, use a fresh cache directory
GT4PY_BUILD_CACHE_DIR=amd_rocm72_repro_$(hostname -s) \
    sbatch --nodelist=nid002510 \
    --output=gt4py_repro.out --error=gt4py_repro.err \
    amd_scripts/sbatch_gt4py_timer_rocm72.sh

# wait, then read the timer report
grep -A 4 "GT4Py Timer Report" gt4py_repro.out | tail -2
```

Both scripts use `--exclusive` and set up the right environment. The aac6
script wraps pytest with `run_with_patch.py`, which monkey-patches CuPy's
NVRTC compile to inject `-DHIP_DISABLE_WARP_SYNC_BUILTINS` and prepend
`#include <cupy/hip_workaround.cuh>` — a workaround for the
`__shfl_xor_sync(0xffffffff, ...)` static_assert failure on ROCm 7.2 with
CuPy 14.0.1. The wrapper script lives at `$HOME/run_with_patch.py` on aac6.

## Where the source data lives

On the Mac under `amd_scripts/`:

- `aac6_node{16,26,30}_rocprof/` — full rocprof-compute output (sysinfo, pmc, log) for the 3 aac6 nodes
- `beverin_nid002{510,420}_pmc.csv` — counter data for 2 beverin nodes
- `analyze_*.txt` — rocprof-compute analyze command output, used to cross-check the HBM column
- `node{26,30}_*_summary.txt` — aac6 power traces (-16 trace partial only, errored at compile mid-trace yesterday)
- `nid002{510,920,420}_*_summary.txt` — beverin power traces (3 nodes)
- `env_*.txt` — full per-node environment captures (firmware, driver, kernel, etc.)
- `gt4py_v{2,3}_nid*.out`, `gt4py_aac6_v{1,2}_*.out` — GT4Py timer outputs from the per-node sweeps

To get a per-kernel table from any pmc_perf.csv:

```bash
python amd_scripts/extract_pmc.py <pmc_perf.csv>
```

The output table has L1↔L2 bandwidth, L2-Fabric bandwidth, HBM bandwidth, L2
hit rate, VGPR count, and block size per kernel. The HBM column agrees with
`rocprof-compute analyze` section 4.1.9 to within ~1% (validated on 5 nodes
across both clusters and rocprof-compute versions 3.4.0 and 4.x).

## Methodology gotchas (read before reproducing)

1. **SLURM node routing is non-deterministic.** Two submissions of the same
   job will land on different nodes if you don't pin with `--nodelist=`. Two
   "identical" runs can give different numbers just because of the chip
   difference. Always pin a node when measuring.
2. **Source state matters more than you'd expect.** A small commit difference
   between two icon4py checkouts (e.g., a new `setdefault` line in
   `model_options.py`) can shift kernel times by tens of μs. Verify both
   clusters are at the same commit before claiming a cluster-level difference.
3. **GT4Py's compile cache must be unique per source change.** Reusing a
   build cache after a source edit can hide perf changes (the cached `.so`
   is what runs, not the new code). Always pass a fresh
   `GT4PY_BUILD_CACHE_DIR=...` when you change source, or wipe the cache.
