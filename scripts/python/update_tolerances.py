#!/usr/bin/env -S uv run -q --frozen --group test python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tighten a JSON tolerance store from recorded 'assert_dallclose' measurements.

Run the tests with 'ICON4PY_RECORD_TOLERANCES=<measurements.jsonl>' to record the measured
differences, then pass the store and the measurement file(s) to this script. Tolerances are only
ever tightened (made smaller): a value is updated when the proposed tolerance -- derived from the
measured difference on deterministic CPU backends -- is smaller than the current one. Tolerances
that are already tight, or that would need to be loosened, are left untouched.

The store is a JSON file mapping 'field -> experiment -> absolute tolerance'.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from icon4py.model.testing import tolerances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=pathlib.Path, required=True, help="JSON tolerance store to update."
    )
    parser.add_argument(
        "measurements",
        type=pathlib.Path,
        nargs="+",
        help="Recorded measurement JSON-lines file(s).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report proposed changes without writing the store."
    )
    args = parser.parse_args()

    store = json.loads(args.store.read_text(encoding="utf-8"))
    aggregated = tolerances.aggregate_max_abs(tolerances.load_measurements(args.measurements))

    changes = []
    for (field, experiment), max_abs in sorted(aggregated.items()):
        # Only manage tolerances that already exist in the store.
        if field not in store or experiment not in store[field]:
            continue
        current = store[field][experiment]
        proposed = tolerances.propose_tolerance(max_abs)
        if proposed < current:
            changes.append((field, experiment, current, proposed, max_abs))
            store[field][experiment] = proposed

    if not changes:
        print("No tolerances to tighten.")
        return

    for field, experiment, current, proposed, max_abs in changes:
        print(f"{field}/{experiment}: {current:g} -> {proposed:g} (measured max diff {max_abs:g})")

    if args.dry_run:
        print("Dry run: store not modified.")
        return

    args.store.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Updated {args.store}.")


if __name__ == "__main__":
    main()
