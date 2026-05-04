#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract per-kernel duration, cache/HBM bandwidth, L2 hit rate from
rocprof-compute pmc_perf.csv.

rocprof-compute uses multi-pass profiling: each kernel is run multiple times with
different counter sets. So TCC counters are non-zero only on the specific row
where their counter set was active. We pick the median non-zero value per
(kernel_name, counter) across all rows.

Three throughput columns measure three different cache planes:

L1↔L2 (TB/s):
    (TCC_READ_sum + TCC_WRITE_sum) * 64 / duration. The upstream plane —
    bytes flowing between L1 and L2 cache. Captures L2 hits + misses.
    Use to assess L2 cache reuse / hit rate impact.

L2Fab (TB/s):
    L2-Fabric read + write+atomic bandwidth. The downstream plane — bytes
    flowing between L2 and the fabric (HBM, remote, Infinity Fabric). Only
    L2 misses + writebacks reach this plane. Same formula as rocprof-compute
    §2.1.23 + §2.1.24 (analysis_configs/gfx942/1700_l2_cache.yaml).

    L2_Fabric_Read_bytes = 128*BUBBLE + 64*(RDREQ - BUBBLE - RDREQ_32B)
                          + 32*RDREQ_32B
    L2_Fabric_Write_bytes = 64*WRREQ_64B + 32*(WRREQ - WRREQ_64B)
    L2Fab(TB/s) = (read_bytes + write_bytes) / duration

HBM (TB/s):
    L2Fab * DRAM-traffic fraction (request-count, not byte-weighted —
    matches rocprof-compute's implicit definition).

    HBM(TB/s) = L2_Fab_Read * (RDREQ_DRAM/RDREQ) + L2_Fab_Write * (WRREQ_DRAM/WRREQ)

    Validated 2026-05-03 against `rocprof-compute analyze -k <kernel_id>`
    on map_100_fieldop_1_0_0 (nid002510): our 2.59 TB/s vs §4.1.9
    HBM Bandwidth 2.601 TB/s — match within 0.4%.

    Atomics are included in WRREQ; uncached requests are included in
    RDREQ/WRREQ. Achieved (not theoretical peak — that's §17.1.5 in newer
    rocprof-compute, ~8.6 TB/s for MI300A).

L2 hit rate formula:
    TCC_HIT_sum / TCC_REQ_sum. Matches rocprof-compute §2.1.21 (47.4% vs
    47.5% on map_100_fieldop_1).
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
            # HBM-related counters from L2-Fabric (TCC_EA0_*).
            "rdreq": [],  # TCC_EA0_RDREQ_sum
            "rdreq_32b": [],  # TCC_EA0_RDREQ_32B_sum
            "rdreq_dram": [],  # TCC_EA0_RDREQ_DRAM_sum
            "wrreq": [],  # TCC_EA0_WRREQ_sum  (includes atomics)
            "wrreq_64b": [],  # TCC_EA0_WRREQ_64B_sum
            "wrreq_dram": [],  # TCC_EA0_WRREQ_DRAM_sum
            "bubble": [],  # TCC_BUBBLE_sum (128B reads)
            "arch_vgpr": "",
            "workgroup": "",
        }
    )

    counter_columns = [
        ("hit", "TCC_HIT_sum"),
        ("req", "TCC_REQ_sum"),
        ("read", "TCC_READ_sum"),
        ("write", "TCC_WRITE_sum"),
        ("rdreq", "TCC_EA0_RDREQ_sum"),
        ("rdreq_32b", "TCC_EA0_RDREQ_32B_sum"),
        ("rdreq_dram", "TCC_EA0_RDREQ_DRAM_sum"),
        ("wrreq", "TCC_EA0_WRREQ_sum"),
        ("wrreq_64b", "TCC_EA0_WRREQ_64B_sum"),
        ("wrreq_dram", "TCC_EA0_WRREQ_DRAM_sum"),
        ("bubble", "TCC_BUBBLE_sum"),
    ]

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
            for key, col in counter_columns:
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

    def _med(vals):
        return statistics.median(vals) if vals else 0

    print(
        f"{'Kernel':<28} {'Dur(us)':>8} {'L1↔L2(TB/s)':>12} {'L2Fab(TB/s)':>12} "
        f"{'HBM(TB/s)':>10} {'L2 Hit%':>8} {'VGPR':>5} {'Block':>6}"
    )
    print("-" * 100)
    sorted_keys = sorted(
        kernels.keys(),
        key=lambda k: -statistics.median(kernels[k]["durs"]) if kernels[k]["durs"] else 0,
    )
    for name in sorted_keys:
        d = kernels[name]
        if not d["durs"]:
            continue
        dur_us = statistics.median(d["durs"]) / 1000
        dur_s = dur_us * 1e-6
        # L1↔L2 plane: TCC_READ + TCC_WRITE * 64B per L2 transaction.
        l1l2_bytes = (_med(d["read"]) + _med(d["write"])) * 64
        l1l2_tbs = l1l2_bytes / dur_s / 1e12 if dur_s > 0 else 0
        hit_pct = 100 * _med(d["hit"]) / _med(d["req"]) if _med(d["req"]) > 0 else 0

        # L2-Fabric plane: rocprof-compute §2.1.23 + §2.1.24 formulas.
        rdreq = _med(d["rdreq"])
        rdreq_32b = _med(d["rdreq_32b"])
        rdreq_dram = _med(d["rdreq_dram"])
        wrreq = _med(d["wrreq"])
        wrreq_64b = _med(d["wrreq_64b"])
        wrreq_dram = _med(d["wrreq_dram"])
        bubble = _med(d["bubble"])
        l2_fab_rd_bytes = 128 * bubble + 64 * max(rdreq - bubble - rdreq_32b, 0) + 32 * rdreq_32b
        l2_fab_wr_bytes = 64 * wrreq_64b + 32 * max(wrreq - wrreq_64b, 0)
        l2_fab_tbs = (l2_fab_rd_bytes + l2_fab_wr_bytes) / dur_s / 1e12 if dur_s > 0 else 0

        # HBM plane: L2-Fabric * DRAM-traffic fraction. Validated against
        # rocprof-compute §4.1.9 within 0.4% (see module docstring).
        rd_dram_frac = rdreq_dram / rdreq if rdreq > 0 else 0
        wr_dram_frac = wrreq_dram / wrreq if wrreq > 0 else 0
        hbm_bytes = l2_fab_rd_bytes * rd_dram_frac + l2_fab_wr_bytes * wr_dram_frac
        hbm_tbs = hbm_bytes / dur_s / 1e12 if dur_s > 0 else 0

        print(
            f"{name[:28]:<28} {dur_us:>8.1f} {l1l2_tbs:>12.2f} {l2_fab_tbs:>12.2f} "
            f"{hbm_tbs:>10.2f} {hit_pct:>8.1f} {d['arch_vgpr']:>5} {d['workgroup']:>6}"
        )


if __name__ == "__main__":
    main()
