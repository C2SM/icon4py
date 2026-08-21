# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3
"""Compile-profiler report from a comprof JSONL log."""

from __future__ import annotations

import collections
import json
import pathlib
import sys


def main(path: str) -> None:  # noqa: PLR0912 [report aggregations are branchy by nature]
    with pathlib.Path(path).open(encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events:
        print("no events")
        return
    t0 = min(e["ts"] for e in events)
    t1 = max(e["ts"] for e in events)
    by_type = collections.Counter(e["event"] for e in events)
    print(f"log: {path}  span={t1 - t0:.0f}s  events={len(events)}")
    print("event counts:", dict(by_type))

    # --- per-program table
    prog = collections.defaultdict(collections.Counter)
    submits = collections.Counter()
    finishes = collections.Counter()
    finish_wait = collections.Counter()
    make_task = collections.Counter()
    for e in events:
        n = e.get("name", "?")
        if e["event"] == "step":
            prog[n][e["phase"]] += e["dur"]
        elif e["event"] == "executor":
            prog[n]["executor_total"] += e["dur"]
        elif e["event"] == "autoopt":
            prog[n]["autoopt"] += e["dur"]
        elif e["event"] == "submit":
            submits[n] += 1
            prog[n]["submit_main"] += e["dur"]
        elif e["event"] == "make_task":
            make_task[n] += e["dur"]
        elif e["event"] == "finish":
            finishes[n] += 1
            finish_wait[n] += e["dur"]
        elif e["event"] == "no_offload":
            prog[n]["no_offload"] += 1

    def total(d):
        return d["translation"] + d["bindings"] + d["compilation"]

    rows = sorted(prog.items(), key=lambda kv: -total(kv[1]))
    print(f"\n=== per-program (top 30 by worker-side sum), {len(rows)} programs ===")
    hdr = f"{'program':<52} {'nsub':>4} {'nfin':>4} {'transl':>8} {'autoopt':>8} {'bind':>6} {'build':>8} {'exec':>8} {'waits':>8}"
    print(hdr)
    sums = collections.Counter()
    for name, d in rows[:30]:
        print(
            f"{name:<52.52} {submits[name]:>4} {finishes[name]:>4} "
            f"{d['translation']:>8.1f} {d['autoopt']:>8.1f} {d['bindings']:>6.1f} "
            f"{d['compilation']:>8.1f} {d['executor_total']:>8.1f} {finish_wait[name]:>8.1f}"
        )
    for name, d in rows:
        for k in ("translation", "autoopt", "bindings", "compilation", "executor_total"):
            sums[k] += d[k]
        sums["submit_main"] += d["submit_main"]
        sums["finish_wait"] += finish_wait[name]
        sums["submits"] += submits[name]
    print(
        f"TOTALS: submits={sums['submits']:.0f} submit_main={sums['submit_main']:.1f}s "
        f"translation={sums['translation']:.1f}s autoopt={sums['autoopt']:.1f}s "
        f"bindings={sums['bindings']:.1f}s build={sums['compilation']:.1f}s "
        f"exec={sums['executor_total']:.1f}s finish_wait={sums['finish_wait']:.1f}s"
    )

    # --- per-test windows
    tests = [e for e in events if e["event"] == "test_start"]
    ends = {(e["pid"], e["nodeid"]): e["ts"] for e in events if e["event"] == "test_end"}
    print(f"\n=== per-test windows ({len(tests)} starts) ===")
    for s in tests:
        end = ends.get((s["pid"], s["nodeid"]), t1)
        a, b = s["ts"], end
        el = b - a
        wsub = [e for e in events if e["event"] == "submit" and a - 1 <= e["ts"] <= b]
        wexec = [e for e in events if e["event"] == "executor" and a - 1 <= e["ts"] <= b]
        wsteps = collections.Counter()
        for e in events:
            if e["event"] == "step" and a - 1 <= e["ts"] <= b:
                wsteps[e["phase"]] += e["dur"]
            if e["event"] == "autoopt" and a - 1 <= e["ts"] <= b:
                wsteps["autoopt"] += e["dur"]
        wfinish = sum(e["dur"] for e in events if e["event"] == "finish" and a - 1 <= e["ts"] <= b)
        print(
            f"{s['nodeid'].split('::')[-1][:80]:<80} wall={el:7.0f}s submits={len(wsub):>3} "
            f"execs={len(wexec):>3} transl={wsteps['translation']:7.0f} auto={wsteps['autoopt']:7.0f} "
            f"build={wsteps['compilation']:6.0f} waits={wfinish:7.0f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/comprof.jsonl")
