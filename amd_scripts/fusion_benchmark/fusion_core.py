# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: PLC0415
"""Controlled fusion comparisons; times are milliseconds per granule call."""

import hashlib
import json
import random
import statistics
from pathlib import Path


THETA = "compute_rho_theta_pgrad_and_update_vn"
SOLVERS = (
    "vertically_implicit_solver_at_predictor_step",
    "vertically_implicit_solver_at_corrector_step",
)
TARGETS = (THETA, *SOLVERS)
COMPARISONS = {
    "combined": ((0, 0), (1, 1)),
    "theta": ((0, 0), (1, 0)),
    "solver": ((0, 0), (0, 1)),
    "solver-increment": ((1, 0), (1, 1)),
}
TARGET = "model/atmosphere/dycore/tests/dycore/integration_tests/test_benchmark_solve_nonhydro.py::test_benchmark_solve_nonhydro[False-False]"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timing_schedule(quartets, seed, placebo=True):
    """Balance both arm orders locally and interleave matched A/A controls."""
    require(quartets >= 4 and quartets % 2 == 0, "Use an even quartet count of at least four.")
    rng = random.Random(seed)
    schedule = []
    for pair in range(quartets // 2):
        orders = ["ABBA", "BAAB"]
        rng.shuffle(orders)
        for offset, order in enumerate(orders):
            phases = ["placebo", "intervention"] if placebo else ["intervention"]
            rng.shuffle(phases)
            schedule.extend((phase, 2 * pair + offset, order) for phase in phases)
    return schedule


def paired_summary(blocks):
    """ABBA blocks are sampling units; preserve interaction and signed effects."""
    require(len(blocks) >= 4 and len(blocks) % 4 == 0, "Expected complete ABBA quartets.")
    contrasts, ratios, orders = [], [], []
    arm_quartets = {"A": [], "B": []}
    for i in range(0, len(blocks), 4):
        group = blocks[i : i + 4]
        require(
            [b["arm"] for b in group] in (["A", "B", "B", "A"], ["B", "A", "A", "B"]),
            "Unbalanced timing order.",
        )
        orders.append("".join(b["arm"] for b in group))
        values = {
            arm: statistics.mean(b["device_ms"] for b in group if b["arm"] == arm)
            for arm in ("A", "B")
        }
        require(values["A"] > 0 and values["B"] > 0, "Nonpositive device time.")
        contrasts.append(values["A"] - values["B"])
        ratios.append(values["B"] / values["A"])
        for arm, samples in arm_quartets.items():
            samples.append(values[arm])
    return dict(
        saved_ms=statistics.mean(contrasts),
        variant_over_baseline=statistics.mean(ratios),
        quartet_saved_ms=contrasts,
        quartet_orders=orders,
        quartet_ratios=ratios,
        quartet_arm_ms=arm_quartets,
        quartet_stdev_ms={
            name: statistics.stdev(values) if len(values) >= 2 else None
            for name, values in dict(arm_quartets, contrast=contrasts).items()
        },
        all_quartets_faster=all(d > 0 for d in contrasts),
    )


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def summarize_metric(report, metric="device_ms", program=None):
    """Report raw and matched-control contrasts without treating calls as replicates."""
    from scipy.stats import t

    def blocks(items):
        return [
            dict(b, device_ms=b[metric] if program is None else b["programs"][program])
            for b in items
        ]

    def interval(values):
        average = statistics.mean(values)
        half = float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
        return [average - half, average + half]

    paired = paired_summary(blocks(report["blocks"]))
    control = paired_summary(blocks(report["placebo"]["blocks"]))
    values, controls = paired["quartet_saved_ms"], control["quartet_saved_ms"]
    require(paired["quartet_orders"] == control["quartet_orders"], "Unmatched control orders.")
    adjusted = [a - b for a, b in zip(values, controls, strict=True)]
    ci, adjusted_ci = interval(values), interval(adjusted)
    threshold = abs(statistics.mean(controls)) + 2 * statistics.stdev(controls)
    groups = {
        order: [v for v, o in zip(values, paired["quartet_orders"], strict=True) if o == order]
        for order in ("ABBA", "BAAB")
    }
    # Welch comparison of orders, as well as opposite signs, detects order sensitivity.
    from scipy.stats import ttest_ind

    variances = sum(statistics.variance(v) for v in groups.values())
    order_p = (
        float(ttest_ind(*groups.values(), equal_var=False).pvalue)
        if variances
        else (1.0 if statistics.mean(groups["ABBA"]) == statistics.mean(groups["BAAB"]) else 0.0)
    )
    order_sensitive = (
        order_p < 0.05 or statistics.mean(groups["ABBA"]) * statistics.mean(groups["BAAB"]) < 0
    )
    resolved = (
        (ci[0] > 0 or ci[1] < 0)
        and abs(paired["saved_ms"]) > threshold
        and (adjusted_ci[0] > 0 if paired["saved_ms"] > 0 else adjusted_ci[1] < 0)
        and not order_sensitive
    )
    return dict(
        baseline_ms=statistics.mean(paired["quartet_arm_ms"]["A"]),
        variant_ms=statistics.mean(paired["quartet_arm_ms"]["B"]),
        saving_percent=100 * paired["saved_ms"] / statistics.mean(paired["quartet_arm_ms"]["A"]),
        saved_ms=paired["saved_ms"],
        ci95_ms=ci,
        placebo_mean_ms=statistics.mean(controls),
        placebo_threshold_ms=threshold,
        control_adjusted_mean_ms=statistics.mean(adjusted),
        control_adjusted_ci95_ms=adjusted_ci,
        saved_ms_by_order={o: statistics.mean(v) for o, v in groups.items()},
        order_sensitive=order_sensitive,
        order_p=order_p,
        status=("resolved improvement" if paired["saved_ms"] > 0 else "resolved regression")
        if resolved
        else "unresolved",
        quartets=paired,
    )


def analyze(report):
    require(
        report["status"] == "complete" and report["validation"]["status"] == "passed",
        "Incomplete or invalid timing run.",
    )
    return dict(
        device=summarize_metric(report),
        wall=summarize_metric(report, "wall_ms"),
        programs={p: summarize_metric(report, program=p) for p in report["blocks"][0]["programs"]},
    )
