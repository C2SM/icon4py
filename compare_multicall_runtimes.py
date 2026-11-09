#!/usr/bin/env python3
"""Compare median compute runtimes between two gt4py_metrics_median.py CSV
outputs, split into two tables:
  - programs called more than once in the application (n_samples > 1 in
    either file)
  - single-shot programs (n_samples == 1 in both files)

Rows are matched between the two files by (name, call_idx) -- NOT
pool_key, since compiled_program_pool_key is not stable across runs/configs
even for the same logical call (verified empirically: the same program+call
has a different pool_key in different sweep-config CSVs).

Example:
    python compare_multicall_runtimes.py baseline_median.csv candidate_median.csv \
        --csv-output comparison.csv
"""
import argparse
import csv
from collections import defaultdict


def load_csv(path):
    """Return {name: {call_idx: (median_float, n_samples)}} for rows with a
    median (gt4py_metrics_median.py already drops no-data rows from the CSV)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))

    by_name_call = defaultdict(dict)
    for r in rows:
        if r["median_compute_s"] != "":
            by_name_call[r["name"]][r["call_idx"]] = (
                float(r["median_compute_s"]), int(r["n_samples"]),
            )
    return by_name_call


def compare(path_a, path_b):
    """Return (matched_multi, missing_multi, matched_single, missing_single),
    where *_multi covers (name, call_idx) pairs with n_samples > 1 in either
    file, and *_single covers pairs with n_samples == 1 in both files."""
    data_a = load_csv(path_a)
    data_b = load_csv(path_b)

    names = set(data_a) | set(data_b)
    call_idxs = defaultdict(set)
    for name in names:
        call_idxs[name].update(data_a.get(name, {}).keys())
        call_idxs[name].update(data_b.get(name, {}).keys())

    matched_multi, missing_multi = [], []
    matched_single, missing_single = [], []
    for name in sorted(names):
        for call_idx in sorted(call_idxs[name], key=int):
            a = data_a.get(name, {}).get(call_idx)
            b = data_b.get(name, {}).get(call_idx)
            n_a = a[1] if a else 0
            n_b = b[1] if b else 0
            is_multi = max(n_a, n_b) > 1
            matched_bucket = matched_multi if is_multi else matched_single
            missing_bucket = missing_multi if is_multi else missing_single

            if a is None or b is None:
                missing_bucket.append(dict(
                    name=name, call_idx=call_idx,
                    median_a=a[0] if a else None, median_b=b[0] if b else None,
                    n_a=n_a, n_b=n_b,
                    status="missing_in_a" if a is None else "missing_in_b",
                ))
            else:
                median_a, median_b = a[0], b[0]
                pct_change = (median_b - median_a) / median_a * 100.0
                matched_bucket.append(dict(
                    name=name, call_idx=call_idx,
                    median_a=median_a, median_b=median_b,
                    n_a=n_a, n_b=n_b,
                    pct_change=pct_change, status="matched",
                ))

    for matched in (matched_multi, matched_single):
        matched.sort(key=lambda r: r["pct_change"], reverse=True)
    for missing in (missing_multi, missing_single):
        missing.sort(key=lambda r: (r["name"], int(r["call_idx"])))

    return matched_multi, missing_multi, matched_single, missing_single


def print_table(title, matched, missing):
    print(f"{title}\n{'-' * len(title)}")
    print(f"{len(matched)} pairs matched in both files, {len(missing)} present in only one file.\n")
    print("Sorted by percent change descending (positive = slower in B, negative = faster in B):\n")
    for r in matched:
        print(f"{r['name']} (call #{r['call_idx']}): "
              f"A={r['median_a']:.6e} s [n={r['n_a']}]  B={r['median_b']:.6e} s [n={r['n_b']}]  "
              f"pct_change={r['pct_change']:+.2f}%")
    if missing:
        print("\nPresent in only one file:\n")
        for r in missing:
            side = "A only" if r["status"] == "missing_in_a" else "B only"
            median = r["median_a"] if r["median_a"] is not None else r["median_b"]
            n = r["n_a"] if r["median_a"] is not None else r["n_b"]
            print(f"{r['name']} (call #{r['call_idx']}): {side}, median={median:.6e} s [n={n}]")
    print()


def write_csv_rows(writer, group_label, matched, missing):
    for r in matched:
        writer.writerow([r["name"], r["call_idx"], group_label, r["median_a"], r["n_a"],
                          r["median_b"], r["n_b"], f"{r['pct_change']:.6f}", r["status"]])
    for r in missing:
        writer.writerow([r["name"], r["call_idx"], group_label,
                          "" if r["median_a"] is None else r["median_a"], r["n_a"],
                          "" if r["median_b"] is None else r["median_b"], r["n_b"],
                          "", r["status"]])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_a", help="First gt4py_metrics_median.py CSV output (baseline).")
    p.add_argument("csv_b", help="Second gt4py_metrics_median.py CSV output (compared against csv_a).")
    p.add_argument("--csv-output", default="multicall_runtime_comparison.csv",
                    help="Path to write the comparison CSV to (default: %(default)s).")
    args = p.parse_args()

    matched_multi, missing_multi, matched_single, missing_single = compare(args.csv_a, args.csv_b)

    print(f"Comparing programs: A={args.csv_a}  B={args.csv_b}\n")
    print_table("Called more than once (n_samples > 1 in either file)", matched_multi, missing_multi)
    print_table("Single-shot programs (n_samples == 1 in both files)", matched_single, missing_single)

    with open(args.csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "call_idx", "call_group", "median_a_s", "n_a", "median_b_s", "n_b",
                          "pct_change", "status"])
        write_csv_rows(writer, "n>1", matched_multi, missing_multi)
        write_csv_rows(writer, "n=1", matched_single, missing_single)

    total = len(matched_multi) + len(missing_multi) + len(matched_single) + len(missing_single)
    print(f"Wrote {total} rows to {args.csv_output}")


if __name__ == "__main__":
    main()
