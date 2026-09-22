# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Reference paired compiler comparison using pytest-benchmark and GT4Py metrics.

Historical causal experiments retain their own analysis; new paired compiler
comparisons use this module's sampling protocol and statistics.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import numbers
import operator
import random
import socket
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gt4py.next.instrumentation import metrics
from scipy.stats import t

from icon4py.model.common.utils import data_allocation


def timing_schedule(quartets: int = 12, seed: int = 20260915) -> list[tuple[str, int, str]]:
    """Balance adjacent ABBA/BAAB quartets and interleave identical-code controls."""
    if quartets < 4 or quartets % 2:
        raise ValueError("Use an even quartet count of at least four.")
    rng = random.Random(seed)
    schedule: list[tuple[str, int, str]] = []
    for pair in range(quartets // 2):
        orders = ["ABBA", "BAAB"]
        rng.shuffle(orders)
        for offset, order in enumerate(orders):
            phases = ["placebo", "intervention"]
            rng.shuffle(phases)
            schedule.extend((phase, 2 * pair + offset, order) for phase in phases)
    return schedule


def _interval(a: list[float], b: list[float]) -> list[float]:
    terms = [statistics.variance(v) / len(v) for v in (a, b)]
    variance = sum(terms)
    denominator = sum(x**2 / (len(v) - 1) for x, v in zip(terms, (a, b), strict=True))
    critical = float(t.ppf(0.975, variance**2 / denominator)) if denominator else 1.96
    effect = statistics.mean(a) - statistics.mean(b)
    half = critical * math.sqrt(variance)
    return [effect - half, effect + half]


def summarize(
    blocks: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Use quartets, not individual calls, as the statistical sampling units."""

    def contrasts(
        items: list[dict[str, Any]],
    ) -> tuple[dict[str, list[float]], list[float], list[str]]:
        arms: dict[str, list[float]] = {"A": [], "B": []}
        effects: list[float] = []
        orders: list[str] = []
        if len(items) < 16 or len(items) % 4:
            raise ValueError("Expected complete balanced quartets.")
        for start in range(0, len(items), 4):
            group = items[start : start + 4]
            order = "".join(x["arm"] for x in group)
            if order not in {"ABBA", "BAAB"}:
                raise ValueError("Unbalanced timing order.")
            orders.append(order)
            for arm, values in arms.items():
                values.append(statistics.mean(value(x) for x in group if x["arm"] == arm))
            effects.append(arms["A"][-1] - arms["B"][-1])
        return arms, effects, orders

    arms, effects, orders = contrasts(blocks)
    _, placebo, control_orders = contrasts(controls)
    if control_orders != orders:
        raise ValueError("Control quartets do not match the intervention order.")
    baseline, variant = (statistics.mean(arms[arm]) for arm in ("A", "B"))
    if baseline <= 0 or variant <= 0:
        raise ValueError("Nonpositive timing measurement.")
    saving = statistics.mean(effects)
    ci = _interval(effects, [0.0] * len(effects))
    adjusted = _interval(
        [a - b for a, b in zip(effects, placebo, strict=True)], [0.0] * len(effects)
    )
    threshold = abs(statistics.mean(placebo)) + 2 * statistics.stdev(placebo)
    by_order = {
        order: [v for v, o in zip(effects, orders, strict=True) if o == order]
        for order in ("ABBA", "BAAB")
    }
    if min(map(len, by_order.values())) < 2:
        raise ValueError("Both quartet orders need at least two observations.")
    order_ci = _interval(by_order["ABBA"], by_order["BAAB"])
    order_sensitive = (
        order_ci[0] > 0
        or order_ci[1] < 0
        or math.prod(statistics.mean(v) for v in by_order.values()) <= 0
    )
    resolved = (ci[0] > 0 or ci[1] < 0) and abs(saving) > threshold and not order_sensitive
    return dict(
        baseline_ms=baseline,
        variant_ms=variant,
        saved_ms=saving,
        reduction_pct=100 * saving / baseline,
        ci95_ms=ci,
        control_adjusted_ci95_ms=adjusted,
        control_bias_ms=statistics.mean(placebo),
        control_threshold_ms=threshold,
        quartet_saved_ms=effects,
        saved_ms_by_order={k: statistics.mean(v) for k, v in by_order.items()},
        order_difference_ci95_ms=order_ci,
        order_sensitive=order_sensitive,
        status=("resolved improvement" if saving > 0 else "resolved regression")
        if resolved
        else "unresolved",
    )


def _inventory(
    roots: list[Any],
) -> tuple[dict[tuple[Any, ...], Any], list[tuple[Any, str, Any]], list[tuple[Any, str, Any]]]:
    visited: set[int] = set()
    arrays: dict[tuple[Any, ...], Any] = {}
    scalars: list[tuple[Any, str, Any]] = []
    scalar_arrays: list[tuple[Any, str, Any]] = []

    def visit(value: Any) -> None:  # noqa: PLR0912 [too-many-branches] -- traverse distinct model container types.
        if id(value) in visited:
            return
        visited.add(id(value))
        if data_allocation.is_ndarray(value):
            pointer = (
                value.__array_interface__["data"][0]
                if isinstance(value, np.ndarray)
                else value.data.ptr
            )
            arrays.setdefault((pointer, value.shape, value.strides, str(value.dtype)), value)
        elif hasattr(value, "ndarray"):
            visit(value.ndarray)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, functools.partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        elif inspect.ismethod(value):
            visit(value.__self__)
        elif inspect.isfunction(value):
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
        elif not isinstance(value, type) and type(value).__module__.startswith("icon4py.model."):
            attributes = (
                {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
                if dataclasses.is_dataclass(value)
                else {}
            )
            attributes.update(vars(value) if hasattr(value, "__dict__") else {})
            for name, item in attributes.items():
                if data_allocation.is_rank0_ndarray(item):
                    # Model reductions may replace a scalar array each call.
                    # Track its attribute, not the lifetime of its buffer.
                    scalar_arrays.append((value, name, item))
                    continue
                if type(value).__module__.startswith("icon4py.model.") and (
                    item is None
                    or isinstance(item, (str, bytes, bool, numbers.Number, enum.Enum, np.generic))
                ):
                    scalars.append((value, name, item))
                visit(item)

    for root in roots:
        visit(root)
    if not arrays:
        raise ValueError("No model arrays found for paired restoration.")
    return arrays, scalars, scalar_arrays


class ModelSnapshot:
    """Restore fields and cached scalar state without reallocating model buffers."""

    def __init__(self, roots: list[Any], synchronize: Callable[[], Any]) -> None:
        self.roots, self.synchronize = roots, synchronize
        self.arrays, self.scalars, self.scalar_arrays = _inventory(roots)
        self.saved = [data_allocation.as_numpy(a).copy() for a in self.current_arrays()]
        digest = hashlib.sha256()
        for a in self.saved:
            digest.update(str((a.shape, str(a.dtype))).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        self.sha256 = digest.hexdigest()

    def current_arrays(self) -> list[Any]:
        """Read current scalar attributes as well as the fixed field buffers."""
        return list(self.arrays.values()) + [getattr(o, n) for o, n, _ in self.scalar_arrays]

    def verify_inventory(self) -> None:
        arrays, scalars, scalar_arrays = _inventory(self.roots)
        before_scalars = {(id(o), n) for o, n, _ in self.scalars}
        after_scalars = {(id(o), n) for o, n, _ in scalars}
        before_arrays = {(id(o), n, str(a.dtype)) for o, n, a in self.scalar_arrays}
        after_arrays = {(id(o), n, str(a.dtype)) for o, n, a in scalar_arrays}
        if (
            arrays.keys() != self.arrays.keys()
            or before_scalars != after_scalars
            or before_arrays != after_arrays
        ):
            raise ValueError(
                "Model state inventory changed during the comparison: "
                f"field buffers added={len(arrays.keys() - self.arrays.keys())}, "
                f"removed={len(self.arrays.keys() - arrays.keys())}; "
                f"scalar attributes added={after_scalars - before_scalars}, "
                f"removed={before_scalars - after_scalars}; "
                f"scalar-array attributes added={after_arrays - before_arrays}, "
                f"removed={before_arrays - after_arrays}."
            )

    def restore(self) -> None:
        self.verify_inventory()
        for obj, name, initial in self.scalars:
            current = getattr(obj, name)
            if current is not initial and current != initial:
                setattr(obj, name, initial)
        for obj, name, initial in self.scalar_arrays:
            if getattr(obj, name) is not initial:
                setattr(obj, name, initial)
        for target, saved in zip(self.current_arrays(), self.saved, strict=True):
            if isinstance(target, np.ndarray):
                target[...] = saved
            else:
                target[...] = data_allocation.array_ns(try_cupy=True).asarray(saved)
        self.synchronize()

    def validate(self, reference: list[np.ndarray]) -> dict[str, Any]:
        finite_values, max_error = 0, 0.0
        self.verify_inventory()
        for index, (array, expected) in enumerate(
            zip(self.current_arrays(), reference, strict=True)
        ):
            actual = data_allocation.as_numpy(array)
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            if np.issubdtype(actual.dtype, np.inexact):
                mask = np.isfinite(expected)
                finite_values += int(mask.sum())
                if mask.any():
                    max_error = max(max_error, float(np.max(np.abs(actual[mask] - expected[mask]))))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-11,
                    atol=1e-12,
                    equal_nan=True,
                    err_msg=f"State field {index} differs.",
                )
            else:
                np.testing.assert_array_equal(actual, expected)
        if finite_values == 0:
            raise ValueError("Validation contained no finite floating-point values.")
        return dict(
            fields=len(reference),
            finite_values=finite_values,
            max_abs_error=max_error,
            rtol=1e-11,
            atol=1e-12,
        )


def _metric_offsets() -> dict[Any, int]:
    return {
        (key, name): len(src.metrics[name].samples)
        for key, src in metrics.sources.items()
        for name in ("compute", "total")
        if name in src.metrics
    }


def _program_samples(before: dict[Any, int], metric_name: str = "compute") -> dict[str, float]:
    result: dict[str, float] = {}
    for key, src in metrics.sources.items():
        metric = src.metrics.get(metric_name)
        values = metric.samples[before.get((key, metric_name), 0) :] if metric else []
        if values:
            name = src.metadata["name"]
            result[name] = result.get(name, 0.0) + sum(values) * 1000
    if not result or any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Missing or invalid GT4Py program timings.")
    return result


def provenance(source_files: Sequence[str | Path]) -> dict[str, Any]:
    packages = {}
    for name in (
        "icon4py-common",
        "icon4py-testing",
        "icon4py-atmosphere-dycore",
        "gt4py",
        "dace",
        "pytest-benchmark",
        "numpy",
    ):
        dist = importlib.metadata.distribution(name)
        packages[name] = dict(
            version=dist.version, source=json.loads(dist.read_text("direct_url.json") or "null")
        )
    return dict(
        hostname=socket.gethostname(),
        packages=packages,
        sources={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
    )


def _program_time(block: dict[str, Any], name: str) -> float:
    return float(block["programs"][name])


def compare(
    benchmark: Any,
    callback: Callable[[], Any],
    *,
    select: Callable[[str], None],
    roots: list[Any],
    synchronize: Callable[[], Any],
    metadata: dict[str, Any],
    source_files: Sequence[str | Path],
    samples_per_block: int = 10,
    warmup: int = 5,
) -> None:
    """Run ordinary pytest-benchmark timing with restored, alternating model arms.

    Its aggregate table mixes both arms and controls; the comparison is stored
    under extra_info.comparison in the ordinary benchmark JSON.
    """
    if benchmark.disabled or benchmark.cprofile:
        raise ValueError("Paired measurements require benchmarking enabled and cprofile disabled.")
    if samples_per_block < 1 or warmup < 1:
        raise ValueError("Warmup and measured-call counts must be positive.")
    snapshot = ModelSnapshot(roots, synchronize)
    initial_provenance = provenance(source_files)
    validation = None
    for arm in ("A", "B"):
        select(arm)
        snapshot.restore()
        callback()
        synchronize()
        if arm == "A":
            reference = [data_allocation.as_numpy(a).copy() for a in snapshot.current_arrays()]
        else:
            validation = snapshot.validate(reference)
            del reference
    schedule = timing_schedule()
    block_specs = [
        (phase, quartet, order, arm) for phase, quartet, order in schedule for arm in order
    ]
    blocks: list[dict[str, Any]] = []
    offsets: dict[Any, int] = {}
    index = 0

    def setup() -> None:
        nonlocal offsets
        if index % samples_per_block == 0:
            phase, quartet, order, arm = block_specs[index // samples_per_block]
            select("A" if phase == "placebo" else arm)
            snapshot.restore()
            for _ in range(warmup):
                callback()
            blocks.append(dict(phase=phase, quartet=quartet, order=order, arm=arm, samples=[]))
        offsets = _metric_offsets()

    def teardown() -> None:
        nonlocal index
        programs = _program_samples(offsets)
        calls = _program_samples(offsets, "total")
        if blocks[0]["samples"] and programs.keys() != blocks[0]["samples"][0]["programs"].keys():
            raise ValueError("Hot-loop program coverage changed between arms.")
        # Teardown runs after each measured call: the last pytest sample is
        # this call's wall time. Block medians are calculated after timing below.
        blocks[-1]["samples"].append(
            dict(
                programs=programs,
                program_calls_ms=calls,
                device_ms=sum(programs.values()),
                wall_ms=benchmark.stats.stats.data[-1] * 1000,
            )
        )
        index += 1

    try:
        benchmark.pedantic(
            callback,
            setup=setup,
            teardown=teardown,
            rounds=len(block_specs) * samples_per_block,
            iterations=1,
        )
        snapshot.verify_inventory()
        if provenance(source_files) != initial_provenance:
            raise ValueError("Sources or package provenance changed during timing.")
        for block in blocks:
            samples = block["samples"]
            block["programs"] = {
                name: statistics.median(s["programs"][name] for s in samples)
                for name in samples[0]["programs"]
            }
            block["device_ms"] = sum(block["programs"].values())
            block["wall_ms"] = statistics.median(s["wall_ms"] for s in samples)
        intervention = [b for b in blocks if b["phase"] == "intervention"]
        controls = [b for b in blocks if b["phase"] == "placebo"]
        summaries = {
            key: summarize(intervention, controls, operator.itemgetter(key))
            for key in ("device_ms", "wall_ms")
        }
        summaries["programs"] = {
            name: summarize(intervention, controls, functools.partial(_program_time, name=name))
            for name in intervention[0]["programs"]
            if all(
                sum(b["programs"][name] for b in intervention if b["arm"] == arm) > 0
                for arm in ("A", "B")
            )
        }
        benchmark.extra_info["comparison"] = dict(
            schema_version=1,
            status="complete",
            metadata=metadata,
            provenance=initial_provenance,
            initial_state_sha256=snapshot.sha256,
            validation=validation,
            restored_scalars=[
                dict(owner=type(o).__qualname__, name=n, initial_value=str(v))
                for o, n, v in snapshot.scalars
            ],
            restored_scalar_arrays=[
                dict(owner=type(o).__qualname__, name=n, initial_value=saved.item())
                for (o, n, _), saved in zip(
                    snapshot.scalar_arrays, snapshot.saved[len(snapshot.arrays) :], strict=True
                )
            ],
            timing_design=dict(
                quartets=12,
                order_seed=20260915,
                warmup=warmup,
                samples_per_block=samples_per_block,
                schedule=schedule,
            ),
            blocks=blocks,
            summaries=summaries,
        )
        for key in ("device_ms", "wall_ms"):
            s = summaries[key]
            print(
                f"\nPaired {key}: {s['baseline_ms']:.6f} -> {s['variant_ms']:.6f} ms; {s['reduction_pct']:.2f}% reduction; {s['status']}; 95% saving interval {s['ci95_ms']} ms"
            )
    finally:
        select("A")
        snapshot.restore()
