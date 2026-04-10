#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare pickle output files from runs with different rank counts.

Usage:
    python scripts/compare_rank_outputs.py rank1.pkl rank2.pkl rank4.pkl
    python scripts/compare_rank_outputs.py rank1.pkl rank2.pkl
"""

import argparse
import pickle
import sys

import numpy as np


FIELDS_1D = {"cell_lat", "cell_lon", "cell_area", "sfc_pressure"}
FIELDS_2D = {"dz", "z_mc", "temperature", "pressure", "u", "v"}


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def compare_fields(name_a: str, data_a: dict, name_b: str, data_b: dict) -> bool:
    """Compare two datasets field-by-field. Returns True if all match."""
    all_ok = True
    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())

    if keys_a != keys_b:
        print(
            f"  WARNING: field sets differ: {keys_a - keys_b} only in {name_a}, {keys_b - keys_a} only in {name_b}"
        )

    common_keys = sorted(keys_a & keys_b)

    print(
        f"\n{'field':>20} | {'shape_' + name_a:>20} | {'shape_' + name_b:>20} | {'max_abs_diff':>14} | {'max_rel_diff':>14} | {'rmse':>14} | {'match':>6}"
    )
    print("-" * 120)

    for key in common_keys:
        a = data_a[key]
        b = data_b[key]

        if a.shape != b.shape:
            print(
                f"{key:>20} | {a.shape!s:>20} | {b.shape!s:>20} | {'SHAPE MISMATCH':>14} | {'':>14} | {'':>14} | {'FAIL':>6}"
            )
            all_ok = False
            continue

        abs_diff = np.abs(a - b)
        max_abs = np.max(abs_diff)

        # Relative diff: avoid division by zero
        denom = np.maximum(np.abs(a), np.abs(b))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.where(denom > 0, abs_diff / denom, 0.0)
        max_rel = np.max(rel_diff)

        rmse = np.sqrt(np.mean(abs_diff**2))
        exact = max_abs == 0.0
        status = "OK" if exact else "DIFF"

        if not exact:
            all_ok = False

        print(
            f"{key:>20} | {a.shape!s:>20} | {b.shape!s:>20} | "
            f"{max_abs:>14.6e} | {max_rel:>14.6e} | {rmse:>14.6e} | {status:>6}"
        )

        if not exact:
            # Show location of max difference
            idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            print(
                f"{'':>20}   max diff at index {idx}: {name_a}={a[idx]:.10e}, {name_b}={b[idx]:.10e}"
            )

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Compare rank output pickle files.")
    parser.add_argument(
        "files", nargs="+", help="Pickle files to compare (e.g., rank1.pkl rank2.pkl rank4.pkl)"
    )
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Need at least 2 files to compare.")
        sys.exit(1)

    datasets = {}
    for path in args.files:
        name = path.rsplit("/", 1)[-1].replace(".pkl", "")
        print(f"Loading {path} ... ", end="", flush=True)
        datasets[name] = load_pickle(path)
        fields = list(datasets[name].keys())
        first_field = datasets[name][fields[0]]
        print(f"{len(fields)} fields, first field shape: {first_field.shape}")

    names = list(datasets.keys())
    all_ok = True

    # Compare each pair against the first file (reference)
    ref_name = names[0]
    ref_data = datasets[ref_name]

    for other_name in names[1:]:
        print(f"\n{'=' * 120}")
        print(f"Comparing {ref_name} vs {other_name}")
        print(f"{'=' * 120}")
        ok = compare_fields(ref_name, ref_data, other_name, datasets[other_name])
        if ok:
            print("\n  => EXACT MATCH")
        else:
            print("\n  => DIFFERENCES FOUND")
            all_ok = False

    # If more than 2 files, also compare 2nd vs 3rd, etc.
    if len(names) > 2:
        for i in range(1, len(names)):
            for j in range(i + 1, len(names)):
                if i == 0:
                    continue  # already done above
                n1, n2 = names[i], names[j]
                print(f"\n{'=' * 120}")
                print(f"Comparing {n1} vs {n2}")
                print(f"{'=' * 120}")
                ok = compare_fields(n1, datasets[n1], n2, datasets[n2])
                if ok:
                    print("\n  => EXACT MATCH")
                else:
                    print("\n  => DIFFERENCES FOUND")
                    all_ok = False

    print(f"\n{'=' * 120}")
    if all_ok:
        print("RESULT: All comparisons are exact matches.")
    else:
        print("RESULT: Some differences were found (see above for details).")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
