#!/usr/bin/env python3
"""Parse a gt4py_metrics.json file and print the median "compute" runtime
for each program (metadata.name), also writing the same info to a CSV file.
Programs whose compiled instance was invoked more than once (same name,
different call index and/or compiled_program_pool_key -- a distinct
compiled shape) are reported as separate rows, never averaged together.

Example:
    python gt4py_metrics_median.py --input gt4py_metrics.json
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

KEY_RE = re.compile(r'^(?P<name>.+)<(?P<backend>[^>]+)>#(?P<idx>\d+)\[(?P<pool>-?\d+)\]$')


def parse_entries(data):
    """Return a list of dicts: name, backend, call_idx, pool_key, median (or
    None if no compute samples), n_samples."""
    entries = []
    for key, value in data.items():
        m = KEY_RE.match(key)
        if not m:
            sys.exit(f"Could not parse metrics key: {key!r}")
        compute = value.get("metrics", {}).get("compute", [])
        median = statistics.median(compute) if compute else None
        entries.append(dict(
            name=m.group("name"),
            backend=m.group("backend"),
            call_idx=int(m.group("idx")),
            pool_key=m.group("pool"),
            median=median,
            n_samples=len(compute),
        ))
    return entries


def format_label(entry, show_backend):
    label = f"{entry['name']}"
    if show_backend:
        label += f"<{entry['backend']}>"
    label += f" (call #{entry['call_idx']}, pool={entry['pool_key']})"
    return label


def group_by_name(entries):
    by_name = defaultdict(list)
    for e in entries:
        by_name[e["name"]].append(e)
    return by_name


def build_csv_rows(entries):
    """CSV rows: only entries with at least one compute sample, sorted by
    median runtime descending (slowest first)."""
    with_data = [e for e in entries if e["median"] is not None]
    return sorted(with_data, key=lambda e: e["median"], reverse=True)


def write_csv(entries, csv_path):
    by_name = group_by_name(entries)
    rows = build_csv_rows(entries)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "backend", "call_idx", "pool_key", "multi_call",
                          "median_compute_s", "n_samples"])
        for e in rows:
            writer.writerow([
                e["name"], e["backend"], e["call_idx"], e["pool_key"],
                len(by_name[e["name"]]) > 1,
                "" if e["median"] is None else e["median"],
                e["n_samples"],
            ])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="gt4py_metrics.json", help="Path to gt4py_metrics.json (default: %(default)s).")
    p.add_argument("--csv-output", default=None,
                    help="Path to write the CSV report to (default: <input>_median.csv).")
    args = p.parse_args()

    csv_path = args.csv_output or f"{os.path.splitext(args.input)[0]}_median.csv"

    with open(args.input) as f:
        data = json.load(f)

    entries = parse_entries(data)
    show_backend = len({e["backend"] for e in entries}) > 1

    with_data = [e for e in entries if e["median"] is not None]
    without_data = [e for e in entries if e["median"] is None]

    with_data.sort(key=lambda e: e["median"], reverse=True)
    without_data.sort(key=lambda e: (e["name"], e["call_idx"]))

    print(f"Median compute runtime by program ({len(with_data)} with data, "
          f"{len(without_data)} with no samples), slowest first:\n")
    for e in with_data:
        label = format_label(e, show_backend)
        print(f"{label}: {e['median']:.6e} s  [n={e['n_samples']}]")

    if without_data:
        print("\nNo samples:\n")
        for e in without_data:
            label = format_label(e, show_backend)
            print(f"{label}: no samples")

    write_csv(entries, csv_path)
    print(f"\nWrote {len(with_data)} rows (timed stencils only) to {csv_path}")


if __name__ == "__main__":
    main()
