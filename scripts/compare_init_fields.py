#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare initial condition fields across different rank-count runs.

Usage:
    python scripts/compare_init_fields.py run_1rank_dir/ run_4rank_dir/

Each directory should contain init_fields_rank*.pkl files.
The script reconstructs the global arrays from per-rank data
and compares them field-by-field with detailed per-level statistics.
"""

import pickle
import sys
from pathlib import Path

import numpy as np


def load_global_fields(run_dir: Path) -> dict[str, np.ndarray]:
    """Load per-rank init pickles and reconstruct global arrays using global indices.

    When multiple ranks provide values for the same global index (halo overlap),
    only the first rank's value is kept (all ranks should agree for owned entries).
    """
    rank_files = sorted(run_dir.glob("init_fields_rank*.pkl"))
    if not rank_files:
        sys.exit(f"No init_fields_rank*.pkl found in {run_dir}")

    print(f"  Loading {len(rank_files)} rank file(s) from {run_dir}")

    rank_data = []
    for f in rank_files:
        with open(f, "rb") as fh:
            rank_data.append(pickle.load(fh))

    # Detect available fields
    sample = rank_data[0]
    cell_fields = [k for k in ["rho", "theta_v", "exner", "w", "u", "v"] if k in sample]
    edge_fields = [k for k in ["vn"] if k in sample]

    max_cell = max(d["cell_global_index"].max() for d in rank_data) + 1
    max_edge = max(d["edge_global_index"].max() for d in rank_data) + 1

    global_fields = {}
    for field_name in cell_fields + edge_fields:
        is_edge = field_name in edge_fields
        gidx_key = "edge_global_index" if is_edge else "cell_global_index"
        n_global = max_edge if is_edge else max_cell

        ref = rank_data[0][field_name]
        shape = (n_global, ref.shape[1]) if ref.ndim == 2 else (n_global,)
        global_arr = np.full(shape, np.nan, dtype=ref.dtype)

        for d in rank_data:
            gidx = d[gidx_key]
            vals = d[field_name]
            # Only fill entries not yet written (avoids halo overwriting owned values)
            if global_arr.ndim == 2:
                mask = np.isnan(global_arr[gidx, 0])
            else:
                mask = np.isnan(global_arr[gidx])
            global_arr[gidx[mask]] = vals[mask]

        n_missing = np.count_nonzero(np.isnan(global_arr.ravel()))
        if n_missing > 0:
            print(f"  WARNING: {field_name} has {n_missing} unfilled global entries")
        global_fields[field_name] = global_arr

    return global_fields


def compare(dir_a: Path, dir_b: Path, verbose: bool = False):
    print(f"Run A: {dir_a}")
    fields_a = load_global_fields(dir_a)
    print(f"Run B: {dir_b}")
    fields_b = load_global_fields(dir_b)

    header = f"{'Field':<12} {'Shape':<20} {'Identical?':^12} {'Max abs diff':>14} {'Max rel diff':>14} {'# diffs':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    all_identical = True
    for name in fields_a:
        if name not in fields_b:
            print(f"{name:<12} {'MISSING in B':<20}")
            continue

        a = fields_a[name]
        b = fields_b[name]
        if a.shape != b.shape:
            print(f"{name:<12} SHAPE MISMATCH: {a.shape} vs {b.shape}")
            continue

        identical = np.array_equal(a, b)
        if identical:
            print(f"{name:<12} {a.shape!s:<20} {'YES':^12}")
        else:
            all_identical = False
            abs_diff = np.abs(a - b)
            max_abs = abs_diff.max()
            denom = np.maximum(np.abs(a), np.abs(b))
            denom = np.where(denom > 0, denom, 1.0)
            max_rel = (abs_diff / denom).max()
            n_diff = np.count_nonzero(a != b)
            print(
                f"{name:<12} {a.shape!s:<20} {'NO':^12} {max_abs:>14.6e} {max_rel:>14.6e} {n_diff:>10}"
            )

            # Per-level breakdown for 2D fields
            if verbose and a.ndim == 2:
                print(
                    f"  {'Level':>7} {'# diffs':>10} {'Max abs':>14} {'Max rel':>14} {'Value range (A)':>30}"
                )
                for k in range(a.shape[1]):
                    col_a, col_b = a[:, k], b[:, k]
                    col_diff = np.abs(col_a - col_b)
                    n_col = np.count_nonzero(col_a != col_b)
                    if n_col == 0:
                        continue
                    col_max_abs = col_diff.max()
                    col_denom = np.maximum(np.abs(col_a), np.abs(col_b))
                    col_denom = np.where(col_denom > 0, col_denom, 1.0)
                    col_max_rel = (col_diff / col_denom).max()
                    vmin, vmax = col_a.min(), col_a.max()
                    print(
                        f"  {k:>7d} {n_col:>10} {col_max_abs:>14.6e} {col_max_rel:>14.6e} [{vmin:>12.4e}, {vmax:>12.4e}]"
                    )

    print()
    if all_identical:
        print("ALL FIELDS ARE BITWISE IDENTICAL after initialization.")
    else:
        print("DIFFERENCES FOUND in initial fields.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare init field dumps across runs.")
    parser.add_argument("dir_a", type=Path, help="Directory with init_fields_rank*.pkl (run A)")
    parser.add_argument("dir_b", type=Path, help="Directory with init_fields_rank*.pkl (run B)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show per-level breakdown for differing fields"
    )
    args = parser.parse_args()
    compare(args.dir_a, args.dir_b, verbose=args.verbose)
