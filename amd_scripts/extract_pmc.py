#!/usr/bin/env python3
"""Extract per-kernel duration, BW, L2 hit rate from rocprof-compute pmc_perf.csv.

rocprof-compute uses multi-pass profiling: each kernel is run multiple times with
different counter sets. So TCC_EA0_RDREQ, TCC_HIT, etc. are non-zero only on the
specific row where their counter set was active. We pick the max non-zero value
per (kernel_name, counter) across all rows.
"""

import csv
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_pmc.py <pmc_perf.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    # For each kernel, store the SUM of each counter across all dispatch rows.
    # (Each pass produces one row per dispatch; we sum per-counter across dispatches
    # for that pass, then sum across passes is meaningless because each pass is the
    # same total. So we take MAX across passes — each pass measures the same kernel
    # invocation count.)
    kernels = defaultdict(lambda: {
        "durs": [],
        "hit": [],
        "req": [],
        "rd": [],
        "wr": [],
        "arch_vgpr": "",
        "workgroup": "",
    })

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Kernel_Name"]
            if "map" not in name:
                continue
            short = re.sub(r"_\d+$", "", name.split("(")[0])
            d = kernels[short]
            try:
                d["durs"].append(float(row["End_Timestamp"]) - float(row["Start_Timestamp"]))
            except (KeyError, ValueError):
                pass
            for key, col in [
                ("hit", "TCC_HIT_sum"),
                ("req", "TCC_REQ_sum"),
                ("rd", "TCC_EA0_RDREQ_sum"),
                ("wr", "TCC_EA0_WRREQ_sum"),
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

    print(f"{'Kernel':<28} {'Dur(us)':>8} {'HBM(TB/s)':>10} {'L2 Hit%':>8} {'VGPR':>5} {'Block':>10}")
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
        rd = statistics.median(d["rd"]) if d["rd"] else 0
        wr = statistics.median(d["wr"]) if d["wr"] else 0
        hit = statistics.median(d["hit"]) if d["hit"] else 0
        req = statistics.median(d["req"]) if d["req"] else 0
        # 64B cache line on MI300A
        hbm_bytes = (rd + wr) * 64
        # bytes / seconds = bytes/s; dur_us * 1e-6 = seconds; / 1e12 = TB/s
        bw_tbs = hbm_bytes / (dur_us * 1e-6) / 1e12 if dur_us > 0 else 0
        hit_pct = 100 * hit / req if req > 0 else 0
        print(
            f"{name[:28]:<28} {dur_us:>8.1f} {bw_tbs:>10.2f} "
            f"{hit_pct:>8.1f} {d['arch_vgpr']:>5} {d['workgroup']:>10}"
        )


if __name__ == "__main__":
    main()
