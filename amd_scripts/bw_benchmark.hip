#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#define CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Benchmark 1: Simple streaming (1 read + 1 write) — baseline peak BW
// ============================================================================
__global__ void kernel_stream_1(
    const double* __restrict__ a0,
    double* __restrict__ out0,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out0[idx] = a0[idx] * 1.0001;
    }
}

// ============================================================================
// Benchmark 2: Many-array read (20 reads + 4 writes) — matches map_100_1 pattern
// Reads 20 arrays at [cell + K_stride * k], writes 4 arrays.
// All coalesced, no indirection.
// ============================================================================
__global__ __launch_bounds__(256) void kernel_many_arrays_20r4w(
    const double* __restrict__ a0,  const double* __restrict__ a1,
    const double* __restrict__ a2,  const double* __restrict__ a3,
    const double* __restrict__ a4,  const double* __restrict__ a5,
    const double* __restrict__ a6,  const double* __restrict__ a7,
    const double* __restrict__ a8,  const double* __restrict__ a9,
    const double* __restrict__ a10, const double* __restrict__ a11,
    const double* __restrict__ a12, const double* __restrict__ a13,
    const double* __restrict__ a14, const double* __restrict__ a15,
    const double* __restrict__ a16, const double* __restrict__ a17,
    const double* __restrict__ a18, const double* __restrict__ a19,
    double* __restrict__ out0, double* __restrict__ out1,
    double* __restrict__ out2, double* __restrict__ out3,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double sum = a0[idx] + a1[idx] + a2[idx] + a3[idx] + a4[idx]
                   + a5[idx] + a6[idx] + a7[idx] + a8[idx] + a9[idx]
                   + a10[idx] + a11[idx] + a12[idx] + a13[idx] + a14[idx]
                   + a15[idx] + a16[idx] + a17[idx] + a18[idx] + a19[idx];
        out0[idx] = sum * 0.25;
        out1[idx] = sum * 0.50;
        out2[idx] = sum * 0.75;
        out3[idx] = sum * 1.00;
    }
}

// ============================================================================
// Benchmark 3: Many-array with vertical offset (k and k+1 reads)
// Matches the actual pattern: some arrays at [idx], some at [idx] and [idx+stride]
// ============================================================================
__global__ __launch_bounds__(256) void kernel_many_arrays_vertical(
    const double* __restrict__ a0,  const double* __restrict__ a1,
    const double* __restrict__ a2,  const double* __restrict__ a3,
    const double* __restrict__ a4,  const double* __restrict__ a5,
    const double* __restrict__ a6,  const double* __restrict__ a7,
    const double* __restrict__ a8,  const double* __restrict__ a9,
    const double* __restrict__ a10, const double* __restrict__ a11,
    const double* __restrict__ a12, const double* __restrict__ a13,
    const double* __restrict__ a14, const double* __restrict__ a15,
    const double* __restrict__ a16, const double* __restrict__ a17,
    const double* __restrict__ a18, const double* __restrict__ a19,
    double* __restrict__ out0, double* __restrict__ out1,
    double* __restrict__ out2, double* __restrict__ out3,
    int cells, int k_levels, int stride
) {
    int cell = (blockIdx.x * 32) + threadIdx.x;
    int k = (blockIdx.y * 8) + threadIdx.y;
    if (cell < cells && k > 0 && k < k_levels) {
        int idx = cell + stride * k;
        int idx_kp1 = cell + stride * (k + 1);
        // 11 arrays at [cell,k] only
        double v0 = a0[idx] + a1[idx] + a2[idx] + a3[idx] + a4[idx] + a5[idx]
                   + a6[idx] + a7[idx] + a8[idx] + a9[idx] + a10[idx];
        // 5 arrays at [cell,k] and [cell,k+1]
        double v1 = (a11[idx] - a11[idx_kp1])
                   + (a12[idx] - a12[idx_kp1])
                   + (a13[idx] * a13[idx_kp1])
                   + (a14[idx] - a14[idx_kp1])
                   + (a15[idx] * a15[idx_kp1]);
        // 4 remaining reads
        double v2 = a16[idx] + a17[idx] + a18[idx] + a19[idx];
        double sum = v0 + v1 + v2;
        out0[idx] = sum * 0.25;
        out1[idx] = sum * 0.50;
        out2[idx] = sum * 0.75;
        out3[idx] = sum * 1.00;
    }
}

// ============================================================================
// Benchmark 4: Many-array with C2E indirection (the real pattern)
// 17 direct reads + 3 indirect reads via connectivity table + 4 writes
// ============================================================================
__global__ __launch_bounds__(256) void kernel_many_arrays_c2e(
    const double* __restrict__ a0,  const double* __restrict__ a1,
    const double* __restrict__ a2,  const double* __restrict__ a3,
    const double* __restrict__ a4,  const double* __restrict__ a5,
    const double* __restrict__ a6,  const double* __restrict__ a7,
    const double* __restrict__ a8,  const double* __restrict__ a9,
    const double* __restrict__ a10, const double* __restrict__ a11,
    const double* __restrict__ a12, const double* __restrict__ a13,
    const double* __restrict__ a14, const double* __restrict__ a15,
    const double* __restrict__ a16,
    const double* __restrict__ edge_field,  // mass_flux equivalent
    const double* __restrict__ geofac,      // geofac_div equivalent [cell, 3]
    const int* __restrict__ c2e,            // connectivity [3, cells]
    double* __restrict__ out0, double* __restrict__ out1,
    double* __restrict__ out2, double* __restrict__ out3,
    int cells, int k_levels, int cell_stride, int c2e_stride
) {
    int cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    int k = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (cell < cells && k > 0 && k < k_levels) {
        int idx = cell + cell_stride * k;
        int idx_kp1 = cell + cell_stride * (k + 1);

        // 6 arrays at [cell,k] only
        double v0 = a0[idx] + a1[idx] + a2[idx] + a3[idx] + a4[idx] + a5[idx];
        // 5 arrays at [cell,k] and [cell,k+1]
        double v1 = (a6[idx] - a6[idx_kp1])
                   + (a7[idx] * a7[idx_kp1])
                   + (a8[idx] - a8[idx_kp1])
                   + (a9[idx] * a9[idx_kp1])
                   + (a10[idx] - a10[idx_kp1]);
        // 6 more direct reads
        double v2 = a11[idx] + a12[idx] + a13[idx] + a14[idx] + a15[idx] + a16[idx];

        // C2E gather: 3 neighbors
        double div = 0.0;
        for (int nbr = 0; nbr < 3; nbr++) {
            int edge_idx = c2e[c2e_stride * nbr + cell];
            double flux = edge_field[cell_stride * k + edge_idx];
            double weight = geofac[c2e_stride * nbr + cell];
            div += flux * weight;
        }

        double sum = v0 + v1 + v2 + div;
        // ~34 FP64 ops total, matching map_100_1
        out0[idx] = sum * 0.25;
        out1[idx] = (v0 - v1) * 0.5;
        out2[idx] = (v1 + div) / (sum + 1e-20);
        out3[idx] = (v2 - div) * (v0 + 1e-20);
    }
}

// ============================================================================
// Benchmark 7: LDS-cached k+1 arrays — same as #4 but uses shared memory
// for the 5 arrays read at both [k] and [k+1].
// Block: 32 cells × 8 K-levels. Each array needs 32 × 9 doubles in LDS (k..k+8).
// LDS usage: 5 arrays × 32 × 9 × 8 = 11,520 bytes per block (well within 64 KB).
// ============================================================================
__global__ __launch_bounds__(256) void kernel_lds_kp1(
    const double* __restrict__ a0,  const double* __restrict__ a1,   // k-only arrays
    const double* __restrict__ a2,  const double* __restrict__ a3,
    const double* __restrict__ a4,  const double* __restrict__ a5,
    const double* __restrict__ a6,  const double* __restrict__ a7,   // k and k+1 arrays
    const double* __restrict__ a8,  const double* __restrict__ a9,
    const double* __restrict__ a10, const double* __restrict__ a11,
    const double* __restrict__ a12, const double* __restrict__ a13,  // more k-only
    const double* __restrict__ a14, const double* __restrict__ a15,
    const double* __restrict__ a16,
    const double* __restrict__ edge_field,
    const double* __restrict__ geofac,
    const int* __restrict__ c2e,
    double* __restrict__ out0, double* __restrict__ out1,
    double* __restrict__ out2, double* __restrict__ out3,
    int cells, int k_levels, int cell_stride, int c2e_stride
) {
    // 5 arrays × 32 cells × 9 K-levels
    __shared__ double smem[5][9][32];

    int cell = (blockIdx.x * 32) + threadIdx.x;
    int k_base = blockIdx.y * 8;
    int tid_k = threadIdx.y;  // 0..7

    if (cell >= cells) return;

    // Cooperatively load 5 arrays × 9 K-levels into LDS
    // Each thread (32×8 = 256 threads) loads multiple elements
    // We need 5 × 9 = 45 loads per cell, 256 threads handle 32 cells × 45 = 1440 loads
    // Simpler: each of 8 K-threads loads its own k and the k=8 thread loads the extra k+1
    for (int arr = 0; arr < 5; arr++) {
        const double* src;
        switch (arr) {
            case 0: src = a7; break;   // k+1 array 0
            case 1: src = a8; break;   // k+1 array 1
            case 2: src = a9; break;   // k+1 array 2
            case 3: src = a10; break;  // k+1 array 3
            case 4: src = a11; break;  // k+1 array 4
        }
        int k = k_base + 1 + tid_k;  // actual K index (1-based)
        if (k <= k_levels) {
            smem[arr][tid_k][threadIdx.x] = src[cell + cell_stride * k];
        }
        // Last K-thread also loads the k+1 for the top of the block
        if (tid_k == 7) {
            int k_extra = k_base + 1 + 8;
            if (k_extra <= k_levels) {
                smem[arr][8][threadIdx.x] = src[cell + cell_stride * k_extra];
            }
        }
    }
    __syncthreads();

    int k = k_base + 1 + tid_k;
    if (k > 0 && k < k_levels) {
        int idx = cell + cell_stride * k;

        // 6 direct k-only reads
        double v0 = a0[idx] + a1[idx] + a2[idx] + a3[idx] + a4[idx] + a5[idx];

        // 5 arrays from LDS at [k] and [k+1]
        double v1 = (smem[0][tid_k][threadIdx.x] - smem[0][tid_k + 1][threadIdx.x])
                   + (smem[1][tid_k][threadIdx.x] * smem[1][tid_k + 1][threadIdx.x])
                   + (smem[2][tid_k][threadIdx.x] - smem[2][tid_k + 1][threadIdx.x])
                   + (smem[3][tid_k][threadIdx.x] * smem[3][tid_k + 1][threadIdx.x])
                   + (smem[4][tid_k][threadIdx.x] - smem[4][tid_k + 1][threadIdx.x]);

        // 6 more direct reads
        double v2 = a12[idx] + a13[idx] + a14[idx] + a15[idx] + a16[idx] + a6[idx];

        // C2E gather
        double div = 0.0;
        for (int nbr = 0; nbr < 3; nbr++) {
            int edge_idx = c2e[c2e_stride * nbr + cell];
            double flux = edge_field[cell_stride * k + edge_idx];
            double weight = geofac[c2e_stride * nbr + cell];
            div += flux * weight;
        }

        double sum = v0 + v1 + v2 + div;
        out0[idx] = sum * 0.25;
        out1[idx] = (v0 - v1) * 0.5;
        out2[idx] = (v1 + div) / (sum + 1e-20);
        out3[idx] = (v2 - div) * (v0 + 1e-20);
    }
}

// ============================================================================
// Benchmark 6: DaCe-style address computation with many int params
// Mimics the exact indexing pattern: array[((-Cell_range_0) + (K_stride * ((-K_range_0) + k)) + cell)]
// with 56 total arguments (19 pointers + 37 ints)
// ============================================================================
__global__ __launch_bounds__(256) void kernel_dace_style(
    const double* __restrict__ a0,  const double* __restrict__ a1,
    const double* __restrict__ a2,  const double* __restrict__ a3,
    const double* __restrict__ a4,  const double* __restrict__ a5,
    const double* __restrict__ a6,  const double* __restrict__ a7,
    const double* __restrict__ a8,  const double* __restrict__ a9,
    const double* __restrict__ a10, const double* __restrict__ a11,
    const double* __restrict__ a12, const double* __restrict__ a13,
    const double* __restrict__ a14, const double* __restrict__ a15,
    double* __restrict__ out0, double* __restrict__ out1,
    double* __restrict__ out2, double* __restrict__ out3,
    int a0_Cell_range_0, int a0_K_range_0, int a0_K_stride,
    int a1_Cell_range_0, int a1_K_range_0, int a1_K_stride,
    int a2_Cell_range_0, int a2_K_range_0, int a2_K_stride,
    int a3_Cell_range_0, int a3_K_range_0, int a3_K_stride,
    int a4_Cell_range_0, int a4_K_range_0, int a4_K_stride,
    int a5_Cell_range_0, int a5_K_range_0, int a5_K_stride,
    int a6_Cell_range_0, int a6_K_range_0, int a6_K_stride,
    int a7_Cell_range_0, int a7_K_range_0, int a7_K_stride,
    int a8_Cell_range_0, int a8_K_range_0, int a8_K_stride,
    int a9_Cell_range_0, int a9_K_range_0, int a9_K_stride,
    int out0_Cell_range_0, int out0_K_range_0, int out0_K_stride,
    int out1_Cell_range_0, int out1_K_range_0, int out1_K_stride,
    double dtime
) {
    int cell = (blockIdx.x * 32) + 4740 + threadIdx.x;
    int k = (blockIdx.y * 8) + 1 + threadIdx.y;
    if (cell >= 4740 && cell <= 44527 && k >= 1 && k <= 79) {
        // DaCe-style indexing: each array has its own range_0, K_range_0, K_stride
        double v0 = a0[((- a0_Cell_range_0) + (a0_K_stride * ((- a0_K_range_0) + k)) + cell)];
        double v1 = a1[((- a1_Cell_range_0) + (a1_K_stride * ((- a1_K_range_0) + k)) + cell)];
        double v2 = a2[((- a2_Cell_range_0) + (a2_K_stride * ((- a2_K_range_0) + k)) + cell)];
        double v3 = a3[((- a3_Cell_range_0) + (a3_K_stride * ((- a3_K_range_0) + k)) + cell)];
        double v4 = a4[((- a4_Cell_range_0) + (a4_K_stride * ((- a4_K_range_0) + k)) + cell)];
        double v5 = a5[((- a5_Cell_range_0) + (a5_K_stride * ((- a5_K_range_0) + k)) + cell)];
        double v6 = a6[((- a6_Cell_range_0) + (a6_K_stride * ((- a6_K_range_0) + k)) + cell)];
        double v7 = a7[((- a7_Cell_range_0) + (a7_K_stride * ((- a7_K_range_0) + k)) + cell)];
        double v8 = a8[((- a8_Cell_range_0) + (a8_K_stride * ((- a8_K_range_0) + k)) + cell)];
        double v9 = a9[((- a9_Cell_range_0) + (a9_K_stride * ((- a9_K_range_0) + k)) + cell)];
        // k+1 reads
        double v6p = a6[((- a6_Cell_range_0) + (a6_K_stride * ((- a6_K_range_0) + k + 1)) + cell)];
        double v7p = a7[((- a7_Cell_range_0) + (a7_K_stride * ((- a7_K_range_0) + k + 1)) + cell)];
        double v8p = a8[((- a8_Cell_range_0) + (a8_K_stride * ((- a8_K_range_0) + k + 1)) + cell)];
        double v9p = a9[((- a9_Cell_range_0) + (a9_K_stride * ((- a9_K_range_0) + k + 1)) + cell)];
        // more reads
        double v10 = a10[((- a2_Cell_range_0) + (a2_K_stride * ((- a2_K_range_0) + k)) + cell)];
        double v11 = a11[((- a3_Cell_range_0) + (a3_K_stride * ((- a3_K_range_0) + k)) + cell)];
        double v12 = a12[((- a4_Cell_range_0) + (a4_K_stride * ((- a4_K_range_0) + k)) + cell)];
        double v13 = a13[((- a5_Cell_range_0) + (a5_K_stride * ((- a5_K_range_0) + k)) + cell)];
        double v14 = a14[((- a0_Cell_range_0) + (a0_K_stride * ((- a0_K_range_0) + k)) + cell)];
        double v15 = a15[((- a1_Cell_range_0) + (a1_K_stride * ((- a1_K_range_0) + k)) + cell)];

        double sum = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
                   + (v6 - v6p) + (v7 * v7p) + (v8 - v8p) + (v9 * v9p)
                   + v10 + v11 + v12 + v13 + v14 + v15;
        // ~34 ops
        double r0 = sum * dtime;
        double r1 = (v0 - v1) * dtime;
        double r2 = (v2 + v3) / (sum + 1e-20);
        double r3 = (v4 - v5) * (v6 + 1e-20);

        out0[((- out0_Cell_range_0) + (out0_K_stride * ((- out0_K_range_0) + k)) + cell)] = r0;
        out1[((- out1_Cell_range_0) + (out1_K_stride * ((- out1_K_range_0) + k)) + cell)] = r1;
        out2[((- out0_Cell_range_0) + (out0_K_stride * ((- out0_K_range_0) + k)) + cell)] = r2;
        out3[((- out1_Cell_range_0) + (out1_K_stride * ((- out1_K_range_0) + k)) + cell)] = r3;
    }
}

// ============================================================================
// Runner
// ============================================================================
// Timing helper — call between manual warmup+timed loops in main
void print_result(const char* name, float* times, int rounds, size_t total_bytes) {
    std::sort(times, times + rounds);
    float median_ms = times[rounds / 2];
    float mean_ms = 0;
    for (int i = 0; i < rounds; i++) mean_ms += times[i];
    mean_ms /= rounds;
    double bw_gb_s = (total_bytes / 1e9) / (median_ms / 1e3);
    printf("%-45s  median: %7.3f ms  mean: %7.3f ms  BW: %7.1f GB/s  (%.1f%% of 3668)\n",
           name, median_ms, mean_ms, bw_gb_s, bw_gb_s / 3668.0 * 100.0);
}

int main() {
    // Match icon_benchmark_regional dimensions
    const int cells = 39788;
    const int k_levels = 80;
    const int N = cells * k_levels;  // total elements per array
    const int stride = cells;        // K_stride (cells per K-level, assuming unit stride)
    const int warmup = 50;
    const int rounds = 200;

    printf("MI300A Bandwidth Benchmark — matching map_100_1 pattern\n");
    printf("Cells: %d, K-levels: %d, N: %d\n", cells, k_levels, N);
    printf("Warmup: %d, Rounds: %d\n\n", warmup, rounds);

    // Allocate 24 arrays (20 read + 4 write)
    const int num_arrays = 24;
    double* d_arrays[num_arrays];
    for (int i = 0; i < num_arrays; i++) {
        CHECK(hipMalloc(&d_arrays[i], N * sizeof(double)));
        CHECK(hipMemset(d_arrays[i], 0, N * sizeof(double)));
    }

    // C2E connectivity (3 x cells, int32)
    int* d_c2e;
    CHECK(hipMalloc(&d_c2e, 3 * cells * sizeof(int)));
    // Fill with semi-random but realistic connectivity (nearby edges)
    std::vector<int> h_c2e(3 * cells);
    for (int c = 0; c < cells; c++) {
        for (int n = 0; n < 3; n++) {
            // Edge index roughly 1.5x the cell index (67096/44528 ≈ 1.5)
            h_c2e[n * cells + c] = (int)((c * 1.5 + n * 7 + (c % 17)) * 1.0) % (cells * 3 / 2);
        }
    }
    CHECK(hipMemcpy(d_c2e, h_c2e.data(), 3 * cells * sizeof(int), hipMemcpyHostToDevice));

    // Edge field (num_edges x k_levels)
    int num_edges = cells * 3 / 2;  // ~67K
    double* d_edge_field;
    CHECK(hipMalloc(&d_edge_field, num_edges * k_levels * sizeof(double)));
    CHECK(hipMemset(d_edge_field, 0, num_edges * k_levels * sizeof(double)));

    // Geofac (3 x cells, double)
    double* d_geofac;
    CHECK(hipMalloc(&d_geofac, 3 * cells * sizeof(double)));
    CHECK(hipMemset(d_geofac, 0, 3 * cells * sizeof(double)));

    hipEvent_t ev_start, ev_stop;
    CHECK(hipEventCreate(&ev_start));
    CHECK(hipEventCreate(&ev_stop));
    float times[rounds];

    printf("%-45s  %s\n", "Kernel", "Results");
    printf("-------------------------------------------  ----------------------------------------------------------------------\n");

    // Benchmark 1: Simple stream (1R + 1W)
    {
        size_t bytes = (size_t)N * sizeof(double) * 2;
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        for (int i = 0; i < warmup; i++)
            kernel_stream_1<<<grid, block>>>(d_arrays[0], d_arrays[20], N);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_stream_1<<<grid, block>>>(d_arrays[0], d_arrays[20], N);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("1. Stream (1R + 1W)", times, rounds, bytes);
    }

    // Benchmark 2: Many arrays flat (20R + 4W)
    {
        size_t bytes = (size_t)N * sizeof(double) * 24;
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_20r4w<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16], d_arrays[17], d_arrays[18], d_arrays[19],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23], N);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_20r4w<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16], d_arrays[17], d_arrays[18], d_arrays[19],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23], N);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("2. Many arrays flat (20R + 4W)", times, rounds, bytes);
    }

    // Benchmark 3: Many arrays with vertical offsets (25R + 4W)
    {
        size_t bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (20 + 5 + 4);
        dim3 block(32, 8);
        dim3 grid((cells + 31) / 32, (k_levels + 7) / 8);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_vertical<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16], d_arrays[17], d_arrays[18], d_arrays[19],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_vertical<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16], d_arrays[17], d_arrays[18], d_arrays[19],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("3. Many arrays + k+1 offsets (25R + 4W)", times, rounds, bytes);
    }

    // Benchmark 4: Full pattern with C2E indirection
    {
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (17 + 5 + 4);
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + c2e_bytes + geofac_bytes + conn_bytes;
        dim3 block(32, 8);
        dim3 grid((cells + 31) / 32, (k_levels + 7) / 8);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("4. Full pattern + C2E indirection", times, rounds, bytes);
    }

    // ========================================================================
    // Benchmark 5: 56 kernel arguments (matching DaCe arg count)
    // Same computation as #4 but pad to 56 args with dummy ints
    // ========================================================================
    // Benchmark 7: LDS-cached k+1 arrays
    {
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (12 + 4);  // 12 k-only + 4 writes
        size_t lds_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * 5;  // 5 arrays loaded once (k+1 reused from LDS)
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + lds_bytes + c2e_bytes + geofac_bytes + conn_bytes;
        dim3 block(32, 8);
        dim3 grid((cells + 31) / 32, (k_levels + 7) / 8);
        for (int i = 0; i < warmup; i++)
            kernel_lds_kp1<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_lds_kp1<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("7. LDS-cached k+1 arrays (5 arrays in smem)", times, rounds, bytes);
    }

    printf("\n--- Isolating bottleneck ---\n");
    {
        // We can't easily add a new kernel here, so let's test DaCe-style addressing
        // by adding dummy int args. Reuse kernel_many_arrays_c2e with extra padding.
        // Instead, test a different angle: match DaCe's exact grid launch config
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (17 + 5 + 4);
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + c2e_bytes + geofac_bytes + conn_bytes;

        // DaCe uses dim3(1244, 10, 1) x dim3(32, 8, 1)
        dim3 block(32, 8);
        dim3 grid(1244, 10, 1);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("5. Same as #4 with DaCe grid dim3(1244,10)", times, rounds, bytes);
    }

    // Benchmark 6: DaCe-style 56 args + per-array indexing
    {
        // Same data volume as #4 approximately
        // 16 reads at k, 4 reads at k+1, 4 writes = 24 array traversals
        size_t bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (16 + 4 + 4);
        dim3 block(32, 8);
        dim3 grid(1244, 10, 1);  // DaCe's exact grid
        int cr = 4740, kr = 0, ks = cells;  // range_0 and stride values
        for (int i = 0; i < warmup; i++)
            kernel_dace_style<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks,
                cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks,
                cr,kr,ks, cr,kr,ks, 1.0);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_dace_style<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15],
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks,
                cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks, cr,kr,ks,
                cr,kr,ks, cr,kr,ks, 1.0);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("6. DaCe-style (56 args, per-array indexing)", times, rounds, bytes);
    }

    // Benchmark 8: Block size 64x4 (matches MI300A wavefront size of 64)
    {
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (17 + 5 + 4);
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + c2e_bytes + geofac_bytes + conn_bytes;
        dim3 block(64, 4);  // 64 cells x 4 K-levels (256 threads, wavefront-aligned)
        dim3 grid((cells + 63) / 64, (k_levels + 3) / 4);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("8. Block 64x4 (wavefront-aligned for MI300A)", times, rounds, bytes);
    }

    // Benchmark 9: Block size 128x2
    {
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (17 + 5 + 4);
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + c2e_bytes + geofac_bytes + conn_bytes;
        dim3 block(128, 2);
        dim3 grid((cells + 127) / 128, (k_levels + 1) / 2);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("9. Block 128x2", times, rounds, bytes);
    }

    // Benchmark 10: Block size 256x1 (max cell coalescing)
    {
        size_t direct_bytes = (size_t)cells * (k_levels - 1) * sizeof(double) * (17 + 5 + 4);
        size_t c2e_bytes = (size_t)cells * (k_levels - 1) * 3 * sizeof(double);
        size_t geofac_bytes = 3 * cells * sizeof(double);
        size_t conn_bytes = 3 * cells * sizeof(int);
        size_t bytes = direct_bytes + c2e_bytes + geofac_bytes + conn_bytes;
        dim3 block(256, 1);
        dim3 grid((cells + 255) / 256, k_levels);
        for (int i = 0; i < warmup; i++)
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
        CHECK(hipDeviceSynchronize());
        for (int i = 0; i < rounds; i++) {
            CHECK(hipEventRecord(ev_start));
            kernel_many_arrays_c2e<<<grid, block>>>(
                d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3], d_arrays[4],
                d_arrays[5], d_arrays[6], d_arrays[7], d_arrays[8], d_arrays[9],
                d_arrays[10], d_arrays[11], d_arrays[12], d_arrays[13], d_arrays[14],
                d_arrays[15], d_arrays[16],
                d_edge_field, d_geofac, d_c2e,
                d_arrays[20], d_arrays[21], d_arrays[22], d_arrays[23],
                cells, k_levels, stride, cells);
            CHECK(hipEventRecord(ev_stop));
            CHECK(hipEventSynchronize(ev_stop));
            CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
        }
        print_result("10. Block 256x1 (max cell coalescing)", times, rounds, bytes);
    }

    printf("\nReference: MI300A peak HBM BW = 3668 GB/s\n");
    printf("map_100_1 measured: 356.6 MB in 240 us = 1486 GB/s (40.5%%)\n");

    // Cleanup
    for (int i = 0; i < num_arrays; i++) CHECK(hipFree(d_arrays[i]));
    CHECK(hipFree(d_c2e));
    CHECK(hipFree(d_edge_field));
    CHECK(hipFree(d_geofac));

    return 0;
}
