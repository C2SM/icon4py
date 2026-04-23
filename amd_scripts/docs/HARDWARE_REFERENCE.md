[← Back to main report](../PROFILING_RESULTS.md)

# Hardware Reference

Verified against vendor specs and on-node `sysinfo.csv` for both platforms used
in this study.

## MI300A (gfx942, "Antares" CDNA3, AMD)

Sources: AMD MI300A data sheet, ROCm GPU architecture docs, chipsandcheese MI300A
deep dive, on-node `sysinfo.csv` (your specific Beverin node):

| Component                               | Spec                                        | Source                                  |
| --------------------------------------- | ------------------------------------------- | --------------------------------------- |
| HBM3 capacity                           | 128 GB                                      | AMD data sheet                          |
| HBM3 peak BW                            | **5.3 TB/s** total chip                     | AMD data sheet                          |
| HBM3 stacks                             | 8 stacks                                    | chipsandcheese, AMD HotChips 2024       |
| HBM bus width                           | 8192-bit (1024-bit × 8 stacks @ 5.2 Gbps)   | chipsandcheese                          |
| HBM channels (per `sysinfo.csv`)        | 128                                         | sysinfo.csv                             |
| Compute Units                           | 228 (38 CU × 6 XCDs)                        | chipsandcheese, sysinfo.csv             |
| XCDs                                    | 6                                           | chipsandcheese, sysinfo.csv `num_xcd=6` |
| Wavefront width                         | 64                                          | sysinfo.csv `wave_size=64`              |
| **vL1D / TCP (per CU)**                 | **32 KB**                                   | ROCm docs, sysinfo.csv `gpu_l1=32`      |
| L2 / TCC (per XCD)                      | 4 MB                                        | ROCm docs, sysinfo.csv `gpu_l2=4096`    |
| L2 (whole GPU, 6 XCDs)                  | ~24 MB visible (4 MB × 6)                   | derived                                 |
| Infinity Cache (memory-side)            | **256 MB** total chip (2 MB per CS slice)   | chipsandcheese                          |
| FP64 peak                               | 122.6 TFLOPS (matrix), 61.3 TFLOPS (vector) | rocprof-compute analyze §1.1            |
| HBM peak measured (this node, SPX/NPS1) | **3.47 TB/s** (rocprof-compute roofline)    | rocprof-compute analyze §4.1.9          |

⚠️ Note: the **5.3 TB/s** spec is the *chip-wide HBM peak*; rocprof-compute reports
**3.47 TB/s** as the measured peak achievable in our SPX/NPS1 partition mode.
The gap (~35%) is partition overhead + sustained vs theoretical. We use 3.47 TB/s
for "% of peak" calculations because that's what's actually achievable.

## GH200 Grace Hopper Superchip (sm_90, "Hopper" H100, NVIDIA)

Sources: NVIDIA H100 whitepaper, NVIDIA Hopper Tuning Guide, NVIDIA GH200 launch:

| Component                             | Spec                                         | Source                           |
| ------------------------------------- | -------------------------------------------- | -------------------------------- |
| HBM3 capacity (GPU side)              | 96 GB                                        | NVIDIA GH200 newsroom            |
| HBM3 peak BW (GPU side)               | **4.0 TB/s** (4023 GB/s)                     | NVIDIA GH200 newsroom            |
| HBM3 stacks (Hopper)                  | 6 stacks                                     | H100 whitepaper                  |
| HBM bus width                         | 6144-bit                                     | H100 whitepaper                  |
| Streaming Multiprocessors             | 132                                          | H100 whitepaper (full H100)      |
| Warp width                            | 32                                           | NVIDIA standard                  |
| **L1 + shared (per SM, partitioned)** | **256 KB** (combined L1+texture+shared)      | Hopper Tuning Guide              |
| L1 (typical L1-only allocation)       | up to ~232 KB                                | Hopper Tuning Guide              |
| Shared memory (max per SM)            | 228 KB                                       | Hopper Tuning Guide              |
| L2 (whole GPU)                        | **50 MB**                                    | Hopper Tuning Guide, NVIDIA blog |
| FP64 peak                             | 67 TFLOPS (matrix), 34 TFLOPS (vector)       | H100 whitepaper                  |
| HBM peak measured (this node)         | **~4.0 TB/s** (89% utilization on map_100_1) | ncu, this study                  |

**Plus:** GH200 also has a **480 GB LPDDR5X** Grace CPU-side memory pool, coherently
addressable from the GPU at much lower bandwidth — irrelevant for this kernel since
all working data is in HBM.

## Key architectural differences for this kernel

| Aspect                       | MI300A                        | GH200                           | Effect on this kernel                                                                                                  |
| ---------------------------- | ----------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| HBM peak (achievable)        | 3.47 TB/s                     | 4.0 TB/s                        | **GH200 has 15% higher peak BW**                                                                                       |
| HBM saturation (measured)    | 70%                           | 89%                             | **GH200 also uses more of it**                                                                                         |
| L1 per CU/SM                 | 32 KB                         | 256 KB combined                 | MI300A's smaller L1 is faster-per-byte; benefits from `(256,1,1)` cell-blocking                                        |
| L1 caches (count × size)     | 228 × 32 KB = 7.3 MB total L1 | 132 × 256 KB = 33.8 MB total L1 | GH200 has 4-5× more total L1, but partitioned across more SMs                                                          |
| L2                           | ~24 MB (per-XCD × 6)          | 50 MB                           | GH200 has ~2× more L2                                                                                                  |
| Infinity Cache / memory-side | 256 MB                        | none equivalent                 | MI300A has a large memory-side cache, but it doesn't show as a major effect on this kernel (HBM still saturates first) |
| Warp/wavefront               | 64-wide                       | 32-wide                         | (256,1,1) wavefront-aligns on AMD; (32,8) is warp-aligned on NVIDIA                                                    |

**Why GH200 has more HBM bandwidth:** it's not a generation gap (both are HBM3).
It's how the HBM is wired up — NVIDIA dedicates more aggressive HBM stacks at
higher data rates on the GH200 product. AMD's MI300A trades some HBM-only BW
for the unified CPU+GPU memory architecture and the giant 256 MB Infinity Cache.
The 5.3 TB/s chip-wide peak on MI300A is real but split across the 6 XCDs and
shared with the CPU complex; per-GPU achievable is lower.

## Sources

- [AMD Instinct MI300A data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf)
- [AMD MI300 architecture docs (ROCm)](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html)
- [chipsandcheese: Inside the AMD MI300A's memory subsystem](https://chipsandcheese.com/p/inside-the-amd-radeon-instinct-mi300as)
- [AMD HotChips 2024 MI300X presentation](<https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf>)
- [NVIDIA GH200 newsroom (4 TB/s HBM3)](https://nvidianews.nvidia.com/news/gh200-grace-hopper-superchip-with-hbm3e-memory)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA H100 whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
