#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract per-kernel duration, BW, L2 hit rate from rocprof-compute pmc_perf.csv.

rocprof-compute uses multi-pass profiling: each kernel is run multiple times with
different counter sets. So TCC counters are non-zero only on the specific row
where their counter set was active. We pick the median non-zero value per
(kernel_name, counter) across all rows.

HBM BW formula:
    Use TCC_READ_sum + TCC_WRITE_sum (L2 transactions, 64B each). This matches
    rocprof-compute analyze §4.1.9 "HBM Bandwidth" within ~3% on map_100_fieldop_1
    (verified 2026-04-19 against dispatch 11/23/35: this script gives 2.37 TB/s,
    rocprof-compute reports 2.43 TB/s). The previous formula
    (TCC_EA0_RDREQ + TCC_EA0_WRREQ) * 64 was wrong by 1.6x; EA0 counters are
    External Access channel-0 counters that don't aggregate across all L2
    channels and don't represent total HBM-side traffic.

L2 hit rate formula:
    TCC_HIT_sum / TCC_REQ_sum. Verified to match rocprof-compute §2.1.21 exactly
    (47.4% vs 47.5% on map_100_fieldop_1).
"""

import contextlib
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pmc.py <pmc_perf.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    kernels = defaultdict(
        lambda: {
            "durs": [],
            "hit": [],
            "req": [],
            "read": [],
            "write": [],
            "arch_vgpr": "",
            "workgroup": "",
        }
    )

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Kernel_Name"]
            if "map" not in name:
                continue
            short = re.sub(r"_\d+$", "", name.split("(")[0])
            d = kernels[short]
            with contextlib.suppress(KeyError, ValueError):
                d["durs"].append(float(row["End_Timestamp"]) - float(row["Start_Timestamp"]))
            for key, col in [
                ("hit", "TCC_HIT_sum"),
                ("req", "TCC_REQ_sum"),
                ("read", "TCC_READ_sum"),
                ("write", "TCC_WRITE_sum"),
            ]:
                raw = row.get(col, "")
                if not raw:
                    continue
                try:
                    fv = float(raw)
                except ValueError:
                    continue
                if fv > 0:
                    d[key].append(fv)
            if not d["arch_vgpr"]:
                d["arch_vgpr"] = row.get("Arch_VGPR", "")
                d["workgroup"] = row.get("Workgroup_Size", "")

    print(
        f"{'Kernel':<28} {'Dur(us)':>8} {'HBM(TB/s)':>10} {'L2 Hit%':>8} {'VGPR':>5} {'Block':>10}"
    )
    print("-" * 80)
    sorted_keys = sorted(
        kernels.keys(),
        key=lambda k: -statistics.median(kernels[k]["durs"]) if kernels[k]["durs"] else 0,
    )
    for name in sorted_keys:
        d = kernels[name]
        if not d["durs"]:
            continue
        dur_us = statistics.median(d["durs"]) / 1000
        # Take median of non-zero values for each counter
        read = statistics.median(d["read"]) if d["read"] else 0
        write = statistics.median(d["write"]) if d["write"] else 0
        hit = statistics.median(d["hit"]) if d["hit"] else 0
        req = statistics.median(d["req"]) if d["req"] else 0
        # 64B cache line on MI300A; TCC_READ + TCC_WRITE = L2 transactions
        hbm_bytes = (read + write) * 64
        bw_tbs = hbm_bytes / (dur_us * 1e-6) / 1e12 if dur_us > 0 else 0
        hit_pct = 100 * hit / req if req > 0 else 0
        print(
            f"{name[:28]:<28} {dur_us:>8.1f} {bw_tbs:>10.2f} "
            f"{hit_pct:>8.1f} {d['arch_vgpr']:>5} {d['workgroup']:>10}"
        )


if __name__ == "__main__":
    main()
