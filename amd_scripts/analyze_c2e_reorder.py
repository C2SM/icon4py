#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze C2E scatter and test edge reordering impact (wavefront-aware).

Models per-wavefront coalescing on the C2E gather instruction.

Two metrics reported per scenario:
1. Cache-line utilization (fill model): of bytes brought into a 64-B cache line,
   how many are eventually consumed across the whole warp.
2. Per-wavefront sector coalescing: when one wavefront issues a single global_load
   on the gather, how many 32-B sectors does the GPU need to fetch? Coalesced
   means N_sectors approaches the minimum N_useful_bytes/32.

Usage:
    python amd_scripts/analyze_c2e_reorder.py [grid_file] [--wave-size N] [--cells-per-block N]
"""

import argparse

import netCDF4 as nc
import numpy as np


def load_grid(path):
    ds = nc.Dataset(path)
    c2e = ds.variables["edge_of_cell"][:].T - 1  # [cell, 3], 0-based
    e2c = ds.variables["adjacent_cell_of_edge"][:].T - 1  # [edge, 2], 0-based
    num_cells = c2e.shape[0]
    num_edges = e2c.shape[0]
    return c2e, e2c, num_cells, num_edges


def measure_cache_line_fill(c2e, label, warp_size, cells_per_warp, start_cell=4740):
    """Cache-line fill model: aggregate over the 3 neighbors per cell."""
    num_cells = c2e.shape[0]
    cls = []
    for start in range(start_cell, num_cells - cells_per_warp, cells_per_warp):
        edges = c2e[start : start + cells_per_warp, :]
        valid = edges.flatten()
        valid = valid[valid >= 0]
        if len(valid) == 0:
            continue
        unique_lines = len(set(idx // 8 for idx in valid))  # 8 doubles per 64-B line
        cls.append(unique_lines)
    cls = np.array(cls)
    useful_doubles = cells_per_warp * 3
    useful_bytes = useful_doubles * 8
    avg_bytes_fetched = cls.mean() * 64
    util = useful_bytes / avg_bytes_fetched * 100
    return util, cls


def measure_wavefront_sector_coalescing(c2e, label, wave_size, cells_per_block, start_cell=4740):
    """Per-wavefront coalescing model.

    The kernel issues, per cell, 3 separate `global_load` instructions for the C2E gather
    (one per neighbor i_C2E_gtx_localdim=0,1,2). At the wavefront level, each of these 3
    instructions is issued by `wave_size` threads simultaneously (each thread = different cell).

    For each such instruction, count distinct 32-B sectors the GPU must fetch.

    Coalescing efficiency = N_useful_sectors / N_actual_sectors
    where N_useful_sectors = ceil(wave_size / 4) (each sector holds 4 doubles)
    """
    num_cells = c2e.shape[0]
    sectors_per_instr = []
    for start in range(start_cell, num_cells - cells_per_block, cells_per_block):
        # Each cell-block has cells_per_block / wave_size wavefronts
        # Each wavefront sees `wave_size` consecutive cells along the cell axis
        n_waves = cells_per_block // wave_size
        for w in range(n_waves):
            wave_start = start + w * wave_size
            wave_cells = c2e[wave_start : wave_start + wave_size, :]  # [wave_size, 3]
            # 3 separate gather instructions; each instruction = one neighbor across all lanes
            for nbr in range(3):
                edges_in_wave = wave_cells[:, nbr]
                valid = edges_in_wave[edges_in_wave >= 0]
                if len(valid) == 0:
                    continue
                # Each `mass_flux[edge_idx]` is a double = 8 bytes. Sectors are 32 bytes
                # → 4 doubles per sector. Sector index = edge_idx // 4
                unique_sectors = len(set(int(idx) // 4 for idx in valid))
                sectors_per_instr.append(unique_sectors)
    sectors_per_instr = np.array(sectors_per_instr)
    # Ideal: wave_size doubles fit in ceil(wave_size / 4) sectors
    ideal_sectors = -(-wave_size // 4)  # ceil division
    actual_sectors = sectors_per_instr.mean()
    coalescing = ideal_sectors / actual_sectors * 100
    return coalescing, sectors_per_instr, ideal_sectors


def report_scenario(c2e, label, wave_size, cells_per_block):
    print(f"\n=== {label} ===")
    util, cls = measure_cache_line_fill(c2e, label, wave_size, cells_per_block)
    print(f"  Cache-line fill model (over whole {cells_per_block}-cell block, all 3 neighbors):")
    print(f"    Cache lines/block: mean {cls.mean():.1f}, ideal {cells_per_block * 3 // 8}")
    print(f"    Cache-line utilization: {util:.1f}%")

    coalescing, sectors, ideal = measure_wavefront_sector_coalescing(
        c2e, label, wave_size, cells_per_block
    )
    print("  Per-wavefront sector coalescing (per single global_load instruction):")
    print(f"    Sectors/instruction: mean {sectors.mean():.1f}, ideal {ideal}")
    print(f"    Per-instruction coalescing: {coalescing:.1f}% of peak")
    return util, coalescing


def reorder_edges_by_min_cell(c2e, num_edges):
    min_cell_per_edge = np.full(num_edges, num_edges, dtype=np.int64)
    for cell_idx in range(c2e.shape[0]):
        for nbr in range(3):
            edge_idx = c2e[cell_idx, nbr]
            if edge_idx >= 0:
                min_cell_per_edge[edge_idx] = min(min_cell_per_edge[edge_idx], cell_idx)
    new_order = np.argsort(min_cell_per_edge)
    old_to_new = np.empty(num_edges, dtype=np.int64)
    old_to_new[new_order] = np.arange(num_edges)
    c2e_new = c2e.copy()
    mask = c2e >= 0
    c2e_new[mask] = old_to_new[c2e[mask]]
    return c2e_new


def reorder_edges_by_cell_block(c2e, num_edges, block_size):
    num_cells = c2e.shape[0]
    old_to_new = np.full(num_edges, -1, dtype=np.int64)
    next_idx = 0
    for start in range(0, num_cells, block_size):
        block_edges = c2e[start : start + block_size, :].flatten()
        block_edges = block_edges[block_edges >= 0]
        unique_edges = np.unique(block_edges)
        for old_idx in unique_edges:
            if old_to_new[old_idx] == -1:
                old_to_new[old_idx] = next_idx
                next_idx += 1
    for i in range(num_edges):
        if old_to_new[i] == -1:
            old_to_new[i] = next_idx
            next_idx += 1
    c2e_new = c2e.copy()
    mask = c2e >= 0
    c2e_new[mask] = old_to_new[c2e[mask]]
    return c2e_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "grid_file",
        nargs="?",
        default="testdata/grids/mch_opr_r19b08/domain1_DOM01.nc",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=64,
        help="Wavefront size: 64 for AMD MI300A, 32 for NVIDIA GH200",
    )
    ap.add_argument(
        "--cells-per-block",
        type=int,
        default=256,
        help="Cells per thread block on Cell axis (256 for our (256,1,1) config)",
    )
    args = ap.parse_args()

    print(f"Grid: {args.grid_file}")
    print(f"Wave size: {args.wave_size}, Cells per block: {args.cells_per_block}")
    print(f"Wavefronts per block: {args.cells_per_block // args.wave_size}")

    c2e, _e2c, num_cells, num_edges = load_grid(args.grid_file)
    print(f"Cells: {num_cells:,}, Edges: {num_edges:,}")

    util_orig, coal_orig = report_scenario(
        c2e, "ORIGINAL (ICON grid generator ordering)", args.wave_size, args.cells_per_block
    )

    c2e_r1 = reorder_edges_by_min_cell(c2e, num_edges)
    util_r1, coal_r1 = report_scenario(
        c2e_r1, "REORDERED: edges sorted by min parent cell", args.wave_size, args.cells_per_block
    )

    c2e_r2 = reorder_edges_by_cell_block(c2e, num_edges, block_size=args.cells_per_block)
    util_r2, coal_r2 = report_scenario(
        c2e_r2,
        f"REORDERED: edges grouped by {args.cells_per_block}-cell blocks",
        args.wave_size,
        args.cells_per_block,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Wavefront sector coalescing (per single global_load):")
    print(f"  Original     : {coal_orig:.1f}%")
    print(f"  Min-cell     : {coal_r1:.1f}% ({coal_r1 - coal_orig:+.1f} pts)")
    print(f"  Block reorder: {coal_r2:.1f}% ({coal_r2 - coal_orig:+.1f} pts)")
    print()
    print("Cache-line fill utilization (over whole block):")
    print(f"  Original     : {util_orig:.1f}%")
    print(f"  Min-cell     : {util_r1:.1f}% ({util_r1 - util_orig:+.1f} pts)")
    print(f"  Block reorder: {util_r2:.1f}% ({util_r2 - util_orig:+.1f} pts)")
    print()
    print(
        "Per-instruction coalescing is the metric ncu/rocprof-compute reports as"
        " 'sector utilization' or 'vL1D coalescing % of peak'. Cache-line fill is"
        " the reuse-over-time metric we previously cited as 85.7%."
    )


if __name__ == "__main__":
    main()
