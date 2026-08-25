#!/usr/bin/env python3
"""Join amd_heuristic SDFG map names (from slurm logs) with rocprof kernel_trace
timings across a sweep of build configs, producing one CSV row per distinct
compiled map version with a total duration column per config and the fastest
config for that version.

By default, compiled versions of the same map that never report a runtime
for the same sweep config (i.e. they never compete for the same
measurement) are merged into a single row, keeping the meta fields of
whichever version was compiled under the canonical baseline config
(CANONICAL_CONFIG below). Pass --no-merge-versions to keep every compiled
version as its own row instead.

Example (regional120):
    python sweep_kernel_timing.py \
        --slurm-log slurm-607212.out --slurm-log slurm-607213.out \
        --trace-regex 'rocprof_.*regional120.*_kernel_trace\.csv$' \
        --output sweep_kernel_timing_regional120.csv

Example (global120):
    python sweep_kernel_timing.py \
        --slurm-log slurm-607214.out --slurm-log slurm-607215.out \
        --trace-regex 'rocprof_.*_global120_.*_kernel_trace\.csv$' \
        --output sweep_kernel_timing_global120.csv
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import pandas as pd

HEUR_RE = re.compile(
    r'\[amd_heuristic\] \[(?P<sdfg>[^\]]+)\]\((?P<mapfull>[^)]+)\): '
    r'vertical=i_K_gtx_vertical\(n=(?P<vn>[0-9A-Za-z]+)\) '
    r'horizontal=i_(?P<hdim>[A-Za-z]+)_gtx_horizontal\(n=(?P<hn>[0-9A-Za-z]+)\) '
    r'indep_bytes=(?P<ib>[0-9A-Za-z]+) total_bytes=(?P<tb>[0-9A-Za-z]+) '
    r'ratio=(?P<ratio>[0-9A-Za-z.]+) tasklets=(?P<tk>[0-9A-Za-z]+)'
)
MAP_PREFIX_RE = re.compile(r'^(?P<prefix>.+customhash__(?P<hash>[0-9a-f]{8})__)')
KERNEL_HASH_RE = re.compile(r'customhash__([0-9a-f]{8})_')
VERSION_SUFFIX_RE = re.compile(r'#v\d+of\d+$')
FIELD_NAMES = ["vertical_n", "horizontal_dim", "horizontal_n",
               "indep_bytes", "total_bytes", "ratio", "tasklets"]

DEFAULT_CONFIG_LABEL_RE = r'(TB2D\[[^\]]+\]_TB1D\[[^\]]+\]_HLB\[[^\]]+\]_VLB\[[^\]]+\])'

# Baseline sweep config whose compiled-version meta fields are kept when
# merging non-conflicting versions of the same map (see merge_compiled_versions).
CANONICAL_CONFIG = "TB2D[256,1,1]_TB1D[256,1,1]_HLB[0]_VLB[0]"


def parse_slurm(path):
    """Return {hash: info-dict} for amd_heuristic lines, skipping any line
    with a None field (unresolved shape info -> not comparable across runs)."""
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            if "amd_heuristic" not in line:
                continue
            m = HEUR_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            if any(d[k] == "None" for k in ("vn", "hn", "ib", "tb", "ratio", "tk")):
                continue
            hm = MAP_PREFIX_RE.match(d["mapfull"])
            if not hm:
                continue
            h = hm.group("hash")
            map_prefix = hm.group("prefix").split("_customhash__")[0]
            out[h] = dict(
                sdfg_name=d["sdfg"], map_prefix=map_prefix,
                vertical_n=d["vn"], horizontal_dim=d["hdim"], horizontal_n=d["hn"],
                indep_bytes=d["ib"], total_bytes=d["tb"], ratio=d["ratio"], tasklets=d["tk"],
            )
    return out


def group_key(info):
    """Identity of a *compiled version* of a map: name + prefix + full shape
    signature. The same (sdfg_name, map_prefix) can compile differently
    (e.g. under loop blocking), so the signature -- not just the name -- is
    what must match for two configs to be considered comparable."""
    return (info["sdfg_name"], info["map_prefix"]) + tuple(info[f] for f in FIELD_NAMES)


def config_label_from_filename(fname, label_re):
    m = re.search(label_re, os.path.basename(fname))
    if not m:
        raise ValueError(f"config-label-regex did not match filename: {fname}")
    return m.group(1)


def find_trace_files(directory, trace_regex):
    pattern = re.compile(trace_regex)
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if pattern.search(f)
    )
    if not files:
        raise SystemExit(f"No files in {directory!r} matched --trace-regex {trace_regex!r}")
    return files


def strip_version_suffix(map_prefix):
    return VERSION_SUFFIX_RE.sub("", map_prefix)


def merge_compiled_versions(rows, configs, canonical_config):
    """Collapse compiled versions of the same (SDFG_NAME, base map_prefix)
    label into one row when none of them ever report a runtime for the same
    sweep config (i.e. they never actually compete for the same
    measurement), keeping the meta fields of whichever version was compiled
    under canonical_config. A label is left split (untouched) if:
      - two or more of its versions report a runtime for the same config
        (a "conflict"), or
      - none of its versions have data for canonical_config.
    """
    canon_col = f"{canonical_config}_total_us"
    groups = defaultdict(list)
    for row in rows:
        groups[(row["SDFG_NAME"], strip_version_suffix(row["map_prefix"]))].append(row)

    merged_rows = []
    n_merged, n_conflict, n_missing_canonical = 0, 0, 0
    for (sdfg_name, base_prefix), group in groups.items():
        if len(group) < 2:
            merged_rows.extend(group)
            continue

        has_conflict = any(
            sum(1 for r in group if r.get(f"{cfg}_total_us", "") != "") >= 2
            for cfg in configs
        )
        if has_conflict:
            n_conflict += 1
            merged_rows.extend(group)
            continue

        canonical_rows = [r for r in group if r.get(canon_col, "") != ""]
        if not canonical_rows:
            n_missing_canonical += 1
            print(f"WARNING: not merging {sdfg_name} / {base_prefix}: "
                  f"no compiled version has data for canonical config {canonical_config!r} "
                  f"({len(group)} versions left split)")
            merged_rows.extend(group)
            continue

        canonical_row = canonical_rows[0]
        merged = {"SDFG_NAME": sdfg_name, "map_prefix": base_prefix}
        merged.update({f: canonical_row[f] for f in FIELD_NAMES})

        best_cfg, best_us, n_present = None, None, 0
        for cfg in configs:
            col = f"{cfg}_total_us"
            present = [r[col] for r in group if r.get(col, "") != ""]
            if present:
                us = present[0]
                merged[col] = us
                n_present += 1
                if best_us is None or us < best_us:
                    best_us, best_cfg = us, cfg
            else:
                merged[col] = ""

        merged["n_hash_instances_total"] = sum(r["n_hash_instances_total"] for r in group)
        merged["n_configs_present"] = n_present
        merged["total_dispatch_count"] = sum(r["total_dispatch_count"] for r in group)
        merged["fastest_config"] = best_cfg
        merged_rows.append(merged)
        n_merged += 1

    print(f"\nMerged compiled versions: {n_merged} labels merged, "
          f"{n_conflict} labels left split (conflicting runtimes), "
          f"{n_missing_canonical} labels left split (no data for canonical config)")
    return merged_rows


def build(slurm_logs, trace_files, config_label_re, out_path, merge_versions=True):
    hash_info = {}
    for p in slurm_logs:
        hash_info.update(parse_slurm(p))
    print(f"Valid (non-None) amd_heuristic hash entries from {slurm_logs}: {len(hash_info)}")

    hash_to_group = {h: group_key(info) for h, info in hash_info.items()}

    by_label_groups = defaultdict(set)
    for gk in hash_to_group.values():
        by_label_groups[gk[:2]].add(gk)

    version_index = {}
    for label, groups in by_label_groups.items():
        for i, gk in enumerate(sorted(groups, key=lambda g: g[2:]), start=1):
            version_index[gk] = i
    n_versions_per_label = {label: len(groups) for label, groups in by_label_groups.items()}

    configs = sorted({config_label_from_filename(f, config_label_re) for f in trace_files})

    agg = defaultdict(dict)  # group_key -> {config: (sum_ns, count)}
    instances_per_group_cfg = defaultdict(lambda: defaultdict(set))

    for fpath in trace_files:
        cfg = config_label_from_filename(fpath, config_label_re)
        df = pd.read_csv(
            fpath,
            usecols=["Kernel_Name", "Start_Timestamp", "End_Timestamp"],
            dtype={"Kernel_Name": "string"},
        )
        df = df[df["Kernel_Name"].str.startswith("map_", na=False)].copy()
        df["hash"] = df["Kernel_Name"].str.extract(KERNEL_HASH_RE)[0]
        df = df.dropna(subset=["hash"])
        df["dur"] = df["End_Timestamp"] - df["Start_Timestamp"]
        df["group_key"] = df["hash"].map(hash_to_group)
        n_unmatched = int(df["group_key"].isna().sum())
        df = df.dropna(subset=["group_key"])

        for h, gk in zip(df["hash"], df["group_key"]):
            instances_per_group_cfg[gk][cfg].add(h)

        grouped = df.groupby("group_key")["dur"].agg(["sum", "count"])
        for gk, row in grouped.iterrows():
            agg[gk][cfg] = (int(row["sum"]), int(row["count"]))
        print(f"{fpath}: {len(grouped)} matched map versions, {n_unmatched} map_* rows unmatched (dropped)")

    rows = []
    for gk, per_cfg in agg.items():
        sdfg_name, map_prefix = gk[0], gk[1]
        meta = dict(zip(FIELD_NAMES, gk[2:]))
        label = (sdfg_name, map_prefix)
        total_versions = n_versions_per_label[label]
        vidx = version_index[gk]
        versioned_map_prefix = map_prefix if total_versions == 1 else f"{map_prefix}#v{vidx}of{total_versions}"

        row = {"SDFG_NAME": sdfg_name, "map_prefix": versioned_map_prefix}
        row.update(meta)

        total_count, best_cfg, best_us, n_present = 0, None, None, 0
        for cfg in configs:
            col = f"{cfg}_total_us"
            if cfg in per_cfg:
                sum_ns, count = per_cfg[cfg]
                us = sum_ns / 1000.0
                row[col] = round(us, 3)
                total_count += count
                n_present += 1
                if best_us is None or us < best_us:
                    best_us, best_cfg = us, cfg
            else:
                row[col] = ""
        row["n_hash_instances_total"] = sum(len(v) for v in instances_per_group_cfg[gk].values())
        row["n_configs_present"] = n_present
        row["total_dispatch_count"] = total_count
        row["fastest_config"] = best_cfg
        rows.append(row)

    if merge_versions:
        rows = merge_compiled_versions(rows, configs, CANONICAL_CONFIG)

    out_df = pd.DataFrame(rows)
    ordered_cols = (
        ["SDFG_NAME", "map_prefix"] + FIELD_NAMES
        + [f"{c}_total_us" for c in configs]
        + ["n_hash_instances_total", "n_configs_present", "total_dispatch_count", "fastest_config"]
    )
    out_df = out_df[ordered_cols].sort_values(["SDFG_NAME", "map_prefix"])
    out_df.to_csv(out_path, index=False)

    n_split = sum(1 for v in n_versions_per_label.values() if v > 1)
    print(f"\nWrote {len(out_df)} rows to {out_path}")
    print(f"Labels split into multiple compiled versions: {n_split} / {len(n_versions_per_label)}")
    print("\nn_configs_present distribution:")
    print(out_df["n_configs_present"].value_counts().sort_index().to_string())
    print("\nfastest_config counts:")
    print(out_df["fastest_config"].value_counts().to_string())
    return out_df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slurm-log", dest="slurm_logs", action="append", required=True,
                    help="Path to a slurm .out file containing [amd_heuristic] lines. Repeatable.")
    p.add_argument("--trace-regex", required=True,
                    help="Regex (re.search) matched against filenames in --dir to select kernel_trace.csv files.")
    p.add_argument("--dir", default=".",
                    help="Directory to search for kernel_trace.csv files (default: cwd).")
    p.add_argument("--config-label-regex", default=DEFAULT_CONFIG_LABEL_RE,
                    help="Regex with one capture group used to derive each file's sweep-config "
                         "label from its filename (default matches TB2D[..]_TB1D[..]_HLB[..]_VLB[..]).")
    p.add_argument("--no-merge-versions", dest="merge_versions", action="store_false", default=True,
                    help="Disable merging compiled map versions that never report a runtime for "
                         "the same sweep config into a single row (merging is enabled by default; "
                         "see CANONICAL_CONFIG in the script).")
    p.add_argument("--output", required=True, help="Output CSV path.")
    args = p.parse_args()

    for p_ in args.slurm_logs:
        if not os.path.isfile(p_):
            sys.exit(f"slurm log not found: {p_}")

    trace_files = find_trace_files(args.dir, args.trace_regex)
    print(f"Matched {len(trace_files)} kernel_trace files:")
    for f in trace_files:
        print(" ", f)

    build(args.slurm_logs, trace_files, args.config_label_regex, args.output,
          merge_versions=args.merge_versions)


if __name__ == "__main__":
    main()
