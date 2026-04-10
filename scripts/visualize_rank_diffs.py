#!/usr/bin/env python3
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize spatial patterns of differences between rank outputs.

Checks for decomposition artifacts: boundary halos, partition-correlated errors, etc.

Usage:
    python scripts/visualize_rank_diffs.py ranks1.pkl ranks2.pkl
    python scripts/visualize_rank_diffs.py ranks1.pkl ranks2.pkl ranks4.pkl
"""

import argparse
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_diff_maps(ref_name, ref_data, other_name, other_data, fields, output_prefix):
    """Plot lat/lon maps of absolute differences for each field."""
    lat = np.rad2deg(ref_data["cell_lat"])
    lon = np.rad2deg(ref_data["cell_lon"])

    for field in fields:
        a = ref_data[field]
        b = other_data[field]
        if a.shape != b.shape:
            continue

        # For 2D fields, pick a few levels
        if a.ndim == 2:
            nlev = a.shape[1]
            levels = [0, nlev // 2, nlev - 1]  # top, mid, bottom
            level_names = ["top", "mid", "bottom"]
        else:
            levels = [None]
            level_names = [""]

        for lev, lev_name in zip(levels, level_names):
            if lev is not None:
                diff = np.abs(a[:, lev] - b[:, lev])
                vals_ref = a[:, lev]
                suffix = f"_lev{lev}_{lev_name}"
            else:
                diff = np.abs(a - b)
                vals_ref = a
                suffix = ""

            if np.max(diff) == 0:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(20, 5))

            # 1) Reference field values
            sc0 = axes[0].scatter(lon, lat, c=vals_ref, s=0.3, cmap="RdBu_r", rasterized=True)
            axes[0].set_title(f"{ref_name}: {field}{suffix}")
            axes[0].set_xlabel("lon (deg)")
            axes[0].set_ylabel("lat (deg)")
            plt.colorbar(sc0, ax=axes[0])

            # 2) Absolute difference
            sc1 = axes[1].scatter(lon, lat, c=diff, s=0.3, cmap="hot_r", rasterized=True)
            axes[1].set_title(f"|{ref_name} - {other_name}|: {field}{suffix}")
            axes[1].set_xlabel("lon (deg)")
            plt.colorbar(sc1, ax=axes[1])

            # 3) Difference with log scale (nonzero only)
            nonzero = diff > 0
            if np.any(nonzero):
                log_diff = np.full_like(diff, np.nan)
                log_diff[nonzero] = np.log10(diff[nonzero])
                sc2 = axes[2].scatter(lon, lat, c=log_diff, s=0.3, cmap="hot_r", rasterized=True)
                axes[2].set_title(f"log10(|diff|): {field}{suffix}")
                axes[2].set_xlabel("lon (deg)")
                plt.colorbar(sc2, ax=axes[2])
            else:
                axes[2].set_title("(no differences)")

            plt.tight_layout()
            fname = f"{output_prefix}_{field}{suffix}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Saved {fname}")


def analyze_diff_histogram(ref_name, ref_data, other_name, other_data, fields, output_prefix):
    """Histogram of cell-wise max absolute diff to see if errors are concentrated."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    idx = 0

    for field in fields:
        a = ref_data[field]
        b = other_data[field]
        if a.shape != b.shape or np.max(np.abs(a - b)) == 0:
            continue
        if idx >= len(axes):
            break

        # Per-cell max diff (over levels if 2D)
        diff = np.abs(a - b)
        if diff.ndim == 2:
            cell_max_diff = np.max(diff, axis=1)
        else:
            cell_max_diff = diff

        nonzero = cell_max_diff[cell_max_diff > 0]
        if len(nonzero) == 0:
            continue

        axes[idx].hist(np.log10(nonzero), bins=50, color="steelblue", edgecolor="k", alpha=0.7)
        axes[idx].set_title(f"{field}: log10(max|diff| per cell)")
        axes[idx].set_xlabel("log10(|diff|)")
        axes[idx].set_ylabel("count")
        n_nonzero = len(nonzero)
        n_total = len(cell_max_diff)
        axes[idx].text(
            0.95,
            0.95,
            f"{n_nonzero}/{n_total} cells\n({100*n_nonzero/n_total:.1f}%)",
            transform=axes[idx].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )
        idx += 1

    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Difference histogram: {ref_name} vs {other_name}", fontsize=14)
    plt.tight_layout()
    fname = f"{output_prefix}_histograms.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def analyze_diff_vs_latitude(ref_data, other_data, fields, ref_name, other_name, output_prefix):
    """Check if differences correlate with latitude (polar artifacts, etc.)."""
    lat = np.rad2deg(ref_data["cell_lat"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    idx = 0

    for field in fields:
        a = ref_data[field]
        b = other_data[field]
        if a.shape != b.shape or np.max(np.abs(a - b)) == 0:
            continue
        if idx >= len(axes):
            break

        diff = np.abs(a - b)
        if diff.ndim == 2:
            cell_max_diff = np.max(diff, axis=1)
        else:
            cell_max_diff = diff

        axes[idx].scatter(lat, cell_max_diff, s=0.3, alpha=0.5, rasterized=True)
        axes[idx].set_title(f"{field}")
        axes[idx].set_xlabel("latitude (deg)")
        axes[idx].set_ylabel("max |diff|")
        idx += 1

    for i in range(idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Diff vs latitude: {ref_name} vs {other_name}", fontsize=14)
    plt.tight_layout()
    fname = f"{output_prefix}_lat_scatter.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rank output differences.")
    parser.add_argument("files", nargs="+", help="Pickle files (first = reference)")
    parser.add_argument("--outdir", default=".", help="Output directory for plots")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Need at least 2 files.")
        sys.exit(1)

    datasets = {}
    for path in args.files:
        name = path.rsplit("/", 1)[-1].replace(".pkl", "")
        print(f"Loading {path} ...")
        datasets[name] = load_pickle(path)

    names = list(datasets.keys())
    dynamic_fields = ["temperature", "pressure", "sfc_pressure", "u", "v"]

    ref_name = names[0]
    ref_data = datasets[ref_name]

    for other_name in names[1:]:
        other_data = datasets[other_name]
        prefix = f"{args.outdir}/diff_{ref_name}_vs_{other_name}"

        print(f"\n=== {ref_name} vs {other_name} ===")

        print("Generating spatial diff maps...")
        plot_diff_maps(ref_name, ref_data, other_name, other_data, dynamic_fields, prefix)

        print("Generating histograms...")
        analyze_diff_histogram(ref_name, ref_data, other_name, other_data, dynamic_fields, prefix)

        print("Generating latitude scatter...")
        analyze_diff_vs_latitude(ref_data, other_data, dynamic_fields, ref_name, other_name, prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
