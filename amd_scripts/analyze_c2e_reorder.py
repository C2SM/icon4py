#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze C2E scatter and test edge reordering impact.

Usage:
    python amd_scripts/analyze_c2e_reorder.py [grid_file]

Default grid file: testdata/grids/mch_opr_r19b08/domain1_DOM01.nc
"""

import sys

import netCDF4 as nc
import numpy as np


def load_grid(path):
    ds = nc.Dataset(path)
    c2e = ds.variables["edge_of_cell"][:].T - 1  # [cell, 3], 0-based
    e2c = ds.variables["adjacent_cell_of_edge"][:].T - 1  # [edge, 2], 0-based
    num_cells = c2e.shape[0]
    num_edges = e2c.shape[0]
    return c2e, e2c, num_cells, num_edges


def measure_scatter(c2e, label, warp_size=32, start_cell=4740):
    """Measure C2E scatter: for each warp of consecutive cells, how spread are the edge indices?"""
    num_cells = c2e.shape[0]
    spreads = []
    cache_lines_per_warp = []

    for start in range(start_cell, num_cells - warp_size, warp_size):
        edges = c2e[start : start + warp_size, :]  # [32, 3]
        all_edges = edges.flatten()
        valid = all_edges[all_edges >= 0]
        if len(valid) == 0:
            continue
        spread = int(valid.max() - valid.min())
        spreads.append(spread)

        # Count unique cache lines accessed (assuming mass_flux[edge] is double, 64B cache line = 8 doubles)
        unique_lines = len(set(idx // 8 for idx in valid))
        cache_lines_per_warp.append(unique_lines)

    spreads = np.array(spreads)
    cls = np.array(cache_lines_per_warp)

    # Useful loads per warp: 32 cells * 3 neighbors = 96 doubles = 96 * 8 = 768 bytes
    useful_bytes = 96 * 8
    wasted_bytes = cls.mean() * 64 - useful_bytes

    print(f"\n=== {label} ===")
    print("Edge index spread per warp (32 cells):")
    print(f"  min:    {spreads.min():>8,}")
    print(f"  median: {int(np.median(spreads)):>8,}")
    print(f"  mean:   {int(spreads.mean()):>8,}")
    print(f"  max:    {spreads.max():>8,}")
    print("Cache lines per warp (3 neighbors, 96 loads):")
    print(f"  min:    {cls.min():>6}")
    print(f"  median: {int(np.median(cls)):>6}")
    print(f"  mean:   {cls.mean():>6.1f}")
    print(f"  max:    {cls.max():>6}")
    print(f"  ideal:  {96 // 8:>6} (if perfectly coalesced, 96 doubles / 8 per line)")
    print(f"Effective cache line utilization: {useful_bytes / (cls.mean() * 64) * 100:.1f}%")
    print(f"Bandwidth amplification: {cls.mean() * 64 / useful_bytes:.2f}x")

    return spreads, cls


def reorder_edges_by_min_cell(c2e, e2c, num_edges):
    """Reorder edges so that edges of nearby cells have nearby indices.

    Strategy: assign each edge to the minimum cell index that uses it,
    then sort edges by that cell index. This ensures edges of cell i
    are near edges of cell i+1.
    """
    # For each edge, find the minimum cell that references it
    min_cell_per_edge = np.full(num_edges, num_edges, dtype=np.int64)

    for cell_idx in range(c2e.shape[0]):
        for nbr in range(3):
            edge_idx = c2e[cell_idx, nbr]
            if edge_idx >= 0:
                min_cell_per_edge[edge_idx] = min(min_cell_per_edge[edge_idx], cell_idx)

    # Sort edges by their minimum parent cell
    new_order = np.argsort(min_cell_per_edge)

    # Create mapping: old_edge_index -> new_edge_index
    old_to_new = np.empty(num_edges, dtype=np.int64)
    old_to_new[new_order] = np.arange(num_edges)

    # Remap C2E table
    c2e_new = c2e.copy()
    mask = c2e >= 0
    c2e_new[mask] = old_to_new[c2e[mask]]

    return c2e_new, old_to_new, new_order


def reorder_edges_by_cell_block(c2e, e2c, num_edges, block_size=32):
    """Reorder edges so that within each cell block, all edges are consecutive.

    Strategy: process cells in blocks of 32 (warp size). Collect all unique
    edges referenced by cells in the block, assign them consecutive new indices.
    Edges shared between blocks get the index from the first block that uses them.
    """
    num_cells = c2e.shape[0]
    old_to_new = np.full(num_edges, -1, dtype=np.int64)
    next_idx = 0

    for start in range(0, num_cells, block_size):
        end = min(start + block_size, num_cells)
        block_edges = c2e[start:end, :].flatten()
        block_edges = block_edges[block_edges >= 0]
        unique_edges = np.unique(block_edges)

        for old_idx in unique_edges:
            if old_to_new[old_idx] == -1:
                old_to_new[old_idx] = next_idx
                next_idx += 1

    # Handle any edges not referenced by any cell
    for i in range(num_edges):
        if old_to_new[i] == -1:
            old_to_new[i] = next_idx
            next_idx += 1

    # Remap C2E table
    c2e_new = c2e.copy()
    mask = c2e >= 0
    c2e_new[mask] = old_to_new[c2e[mask]]

    return c2e_new, old_to_new


def main():
    grid_file = (
        sys.argv[1] if len(sys.argv) > 1 else "testdata/grids/mch_opr_r19b08/domain1_DOM01.nc"
    )
    print(f"Grid: {grid_file}")

    c2e, e2c, num_cells, num_edges = load_grid(grid_file)
    print(f"Cells: {num_cells:,}, Edges: {num_edges:,}")
    print(f"C2E shape: {c2e.shape}, E2C shape: {e2c.shape}")

    # Original ordering
    measure_scatter(c2e, "ORIGINAL (ICON grid generator ordering)")

    # Reorder by min cell
    c2e_reordered1, _, _ = reorder_edges_by_min_cell(c2e, e2c, num_edges)
    measure_scatter(c2e_reordered1, "REORDERED: edges sorted by min parent cell")

    # Reorder by cell block (warp-aligned)
    c2e_reordered2, _ = reorder_edges_by_cell_block(c2e, e2c, num_edges, block_size=32)
    measure_scatter(c2e_reordered2, "REORDERED: edges grouped by 32-cell blocks (warp-aligned)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Potential bandwidth savings from edge reordering")
    print("=" * 70)
    _, cls_orig = (
        measure_scatter.__wrapped__(c2e, "orig")
        if hasattr(measure_scatter, "__wrapped__")
        else (None, None)
    )
    # Re-measure to get numbers
    s1, c1 = measure_scatter(c2e, "Original (recap)")
    s2, c2 = measure_scatter(c2e_reordered1, "Min-cell reorder (recap)")
    s3, c3 = measure_scatter(c2e_reordered2, "Block-32 reorder (recap)")

    print("\nFor the hottest kernel (map_100_1, 240 μs):")
    print(f"C2E loads: 3 per cell, 39788 cells x 79 K-levels = {39788 * 79 * 3:,} total")
    for label, cls in [("Original", c1), ("Min-cell", c2), ("Block-32", c3)]:
        total_cls = cls.sum() * 79  # scale by K-levels (each warp does 1 K-level in the 32x8 block)
        total_bytes = total_cls * 64
        useful = 39788 * 79 * 3 * 8
        print(
            f"  {label:12s}: {total_bytes / 1e6:>7.1f} MB HBM for C2E ({useful / total_bytes * 100:.1f}% useful)"
        )


if __name__ == "__main__":
    main()
