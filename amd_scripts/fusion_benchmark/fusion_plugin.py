# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: PLC0415
"""Benchmark the installed PR pair through ordinary model configuration.

Only dispatch, state restoration, measurement and read-only build capture live
here. The model equations, compiler passes and generated code are not rewritten.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import math
import numbers
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import pytest
from fusion_core import (
    COMPARISONS,
    TARGET,
    TARGETS,
    THETA,
    analyze,
    paired_summary,
    require,
    save_json,
    sha256,
    timing_schedule,
)
from pytest_benchmark.fixture import BenchmarkFixture


STATE = dict(arm="A", build=None, programs={}, patches=[], audits=[], options=[])


def case():
    return json.loads(os.environ["FUSION_CASE"])


@contextlib.contextmanager
def arm_options(arm):
    values = COMPARISONS[case()["name"]][arm == "B"]
    keys = ("ICON4PY_DACE_THETA_FUSION", "ICON4PY_DACE_SOLVER_FUSION")
    saved = {key: os.environ.get(key) for key in keys}
    for key, value in zip(keys, values, strict=True):
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ProgramPair:
    def __init__(self, setup, kwargs):
        self.name = kwargs["program"].__name__
        self.kwargs, self.setup = kwargs, setup
        self.implementations = {}
        self.built = set()
        STATE["programs"][self.name] = self
        with arm_options("A"):
            self.implementations["A"] = setup(**kwargs)

    def __call__(self, *args, **kwargs):
        arm = STATE["arm"] if self.name in TARGETS else "A"
        if arm in self.built:
            return self.implementations[arm](*args, **kwargs)
        STATE["build"] = (self.name, arm)
        try:
            with arm_options(arm):
                if arm not in self.implementations:
                    self.implementations[arm] = self.setup(**self.kwargs)
                result = self.implementations[arm](*args, **kwargs)
                # Variants can compile at a later invocation too; the read-only
                # hook also finds the owner from the SDFG name in that case.
                self.built.add(arm)
                return result
        finally:
            STATE["build"] = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    from dace.codegen import compiler
    from gt4py.next import config as gt_config

    from icon4py.model.common import model_options

    require(
        gt_config.BUILD_JOBS <= 0 or gt_config.BUILD_JOBS_MODE is gt_config.BuildJobsMode.SERIAL,
        "Use serial compilation.",
    )
    require(
        os.environ.get("GT4PY_COLLECT_METRICS_LEVEL") == "10", "Device metrics level must be 10."
    )
    native_setup, native_options = model_options.setup_program, model_options.get_dace_options
    native_folder = compiler.generate_program_folder

    def setup(**kwargs):
        return ProgramPair(native_setup, kwargs)

    def options(program_name, backend_config, **descriptor):
        result = native_options(program_name, backend_config, **descriptor)
        if program_name in TARGETS:
            STATE["options"].append(
                dict(
                    program=program_name,
                    flags=[
                        os.environ.get(k, "0")
                        for k in ("ICON4PY_DACE_THETA_FUSION", "ICON4PY_DACE_SOLVER_FUSION")
                    ],
                    optimization_args=repr(result.get("optimization_args")),
                    scan_fusion=result.get("optimization_args", {}).get("fuse_scan_inputs", False),
                    scan_scope=result.get("optimization_args", {}).get("scan_fusion_scope"),
                )
            )
        return result

    def folder(sdfg, code_objects, *args, **kwargs):
        owner = next((p for p in TARGETS if sdfg.name.startswith(p)), None)
        if owner:
            arm = STATE["build"][1] if STATE["build"] else STATE["arm"]
            destination = (
                Path(os.environ["FUSION_OUTPUT"])
                / "generated"
                / arm
                / f"{owner}_{len(STATE['audits']):03d}"
            )
            destination.mkdir(parents=True)
            copy.deepcopy(sdfg).save(str(destination / "optimized.sdfg"), hash=False, readable=True)
            files = []
            for index, obj in enumerate(code_objects):
                path = destination / f"{index}_{Path(obj.name).name}.{obj.language}"
                path.write_text(obj.code)
                files.append(
                    dict(
                        file=str(path),
                        sha256=sha256(obj.code),
                        kernels=obj.code.count("__global__"),
                    )
                )
            STATE["audits"].append(
                dict(
                    program=owner,
                    arm=arm,
                    build_path=str(args[0] if args else kwargs["out_path"]),
                    files=files,
                )
            )
        return native_folder(sdfg, code_objects, *args, **kwargs)

    for obj, name, replacement in (
        (model_options, "setup_program", setup),
        (model_options, "get_dace_options", options),
        (compiler, "generate_program_folder", folder),
    ):
        STATE["patches"].append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)
    for module in list(sys.modules.values()):
        if (
            module
            and getattr(module, "__name__", "").startswith("icon4py.")
            and getattr(module, "setup_program", None) is native_setup
        ):
            STATE["patches"].append((module, "setup_program", native_setup))
            module.setup_program = setup


def pytest_unconfigure(config):
    for obj, name, value in reversed(STATE["patches"]):
        setattr(obj, name, value)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    config._fusion_completed = False
    require(
        len(items) == 1 and items[0].nodeid.endswith(TARGET),
        "Select exactly the solve_nonhydro granule test.",
    )
    require(not config.getoption("numprocesses", default=0), "Use sequential pytest execution.")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    import numpy as np

    seed = case()["input_seed"]
    sequence = np.random.SeedSequence(seed)
    original = np.random.default_rng
    state = dict(seed=seed, unseeded_rng_calls=0)
    item._fusion_rng = state

    def deterministic(seed=None):
        if seed is None:
            state["unseeded_rng_calls"] += 1
            seed = sequence.spawn(1)[0]
        return original(seed)

    np.random.default_rng = deterministic
    try:
        yield
    finally:
        np.random.default_rng = original


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0 and not getattr(session.config, "_fusion_completed", False):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        print("ERROR: Controlled fusion benchmark did not complete.", flush=True)


def collect_arrays(roots, scalar_state=None):
    """Collect model fields, preserving views and avoiding backend/workspace internals."""
    import cupy as cp
    import numpy as np

    visited, arrays = set(), {}

    def visit(value):  # noqa: PLR0912 -- Traverse model containers and runtime fields.
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, cp.ndarray):
            key = (value.data.ptr, value.shape, value.strides, str(value.dtype))
            arrays.setdefault(key, value)
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
        elif not isinstance(value, type) and (
            dataclasses.is_dataclass(value) or type(value).__module__.startswith("icon4py.model.")
        ):
            attributes = {}
            if dataclasses.is_dataclass(value):
                attributes.update(
                    (f.name, getattr(value, f.name)) for f in dataclasses.fields(value)
                )
            # Include runtime attributes that are not declared dataclass fields.
            attributes.update(vars(value) if hasattr(value, "__dict__") else {})
            if type(value).__module__.startswith("icon4py.model.") and scalar_state is not None:
                for name, item in attributes.items():
                    if item is None or isinstance(
                        item, (str, bytes, bool, numbers.Number, enum.Enum, np.generic)
                    ):
                        scalar_state.append((value, name, item))
            for item in attributes.values():
                visit(item)

    for root in roots:
        visit(root)
    require(arrays, "No model state fields found for paired restoration.")
    return list(arrays.values())


def scalar_record(obj, name, value):
    if isinstance(value, numbers.Real) and not math.isfinite(value):
        value = str(value)
    return dict(owner=type(obj).__qualname__, attribute=name, initial_value=value)


def verify_scalar_inventory(roots, snapshot):
    current = []
    collect_arrays(roots, scalar_state=current)
    before = {(id(obj), name) for obj, name, _ in snapshot}
    after = {(id(obj), name) for obj, name, _ in current}
    require(
        before == after,
        "Primitive model-state inventory changed; new, removed, or retyped attributes "
        "would escape paired restoration. Added: "
        + str(sorted(after - before))
        + "; removed: "
        + str(sorted(before - after)),
    )


def metric_offsets():
    from gt4py.next.instrumentation import metrics

    return {
        key: len(src.metrics.get("compute").samples) if src.metrics.get("compute") else 0
        for key, src in metrics.sources.items()
    }


def device_samples(before):
    from gt4py.next.instrumentation import metrics

    result = {}
    for key, src in metrics.sources.items():
        metric = src.metrics.get("compute")
        samples = metric.samples[before.get(key, 0) :] if metric else []
        if samples:
            name = src.metadata["name"]
            result[name] = result.get(name, 0.0) + sum(samples) * 1000
    require(THETA in result, "Missing theta-rho in-SDFG device timer.")
    require(all(0 <= t < 1e6 for t in result.values()), "Invalid ordinary device timings.")
    return result


class FusionBenchmark(BenchmarkFixture):
    def __init__(self, request):
        self.request = request
        self.name = request.node.name
        self.extra_info = {}
        self.disabled = self.skipped = self.has_error = self._called = False

    def __call__(self, callback, *args, **kwargs):  # noqa: PLR0912, PLR0915 -- Keep the measured restoration/timing sequence together.
        import cupy as cp
        import numpy as np

        require(not self._called, "Benchmark callback invoked twice.")
        self._called = True
        report = dict(
            schema_version=1,
            case=case(),
            hostname=socket.gethostname(),
            platform=case()["platform"],
            grid=self.request.config.getoption("grid"),
            status="running",
            blocks=[],
            gpu_properties=cp.cuda.runtime.getDeviceProperties(0),
            seed_state=self.request.node._fusion_rng,
        )
        output = str(Path(os.environ["FUSION_OUTPUT"]) / "timing.json")
        warmup = case()["warmup"]
        rounds = case()["rounds"]
        try:
            scalar_state = []
            roots = [callback, args, kwargs] + [
                p.kwargs.get("constant_args", {}) for p in STATE["programs"].values()
            ]
            arrays = collect_arrays(roots, scalar_state=scalar_state)
            report["restored_scalar_state"] = [
                scalar_record(obj, name, value) for obj, name, value in scalar_state
            ]
            report["state_arrays"] = [
                dict(shape=a.shape, strides=a.strides, dtype=str(a.dtype), bytes=a.nbytes)
                for a in arrays
            ]
            # Host snapshots avoid doubling GPU allocations, particularly on GH200.
            saved = [cp.asnumpy(a) for a in arrays]
            fingerprint = hashlib.sha256()
            for array in saved:
                fingerprint.update(str((array.shape, str(array.dtype))).encode())
                fingerprint.update(np.ascontiguousarray(array).tobytes())
            report["initial_state_sha256"] = fingerprint.hexdigest()
            report["initial_scalar_state_sha256"] = sha256(
                json.dumps(
                    report["restored_scalar_state"], sort_keys=True, default=str, allow_nan=False
                )
            )
            if hasattr(self.request, "getfixturevalue"):
                mesh = self.request.getfixturevalue("grid_manager").grid
                report["grid_dimensions"] = dict(
                    cells=mesh.num_cells,
                    edges=mesh.num_edges,
                    vertices=mesh.num_vertices,
                    levels=mesh.num_levels,
                    limited_area=mesh.limited_area,
                )

            def restore():
                verify_scalar_inventory(roots, scalar_state)
                for obj, name, value in scalar_state:
                    current = getattr(obj, name)
                    # Leave unchanged frozen dataclass configuration alone.
                    if current is not value and current != value:
                        setattr(obj, name, value)
                for target, source in zip(arrays, saved):
                    target[...] = cp.asarray(source)
                cp.cuda.runtime.deviceSynchronize()

            def invoke():
                return callback(*args, **kwargs)

            # Build both arms before timing/capture. GPU validation runs once in
            # the ordinary timing process, on all collected state fields.
            expected = None
            for arm in ("A", "B"):
                STATE["arm"] = arm
                restore()
                invoke()
                cp.cuda.runtime.deviceSynchronize()
                verify_scalar_inventory(roots, scalar_state)
                if arm == "A":
                    expected = [cp.asnumpy(a) for a in arrays]
                else:
                    finite_values, max_error = 0, 0.0
                    for index, (array, reference) in enumerate(zip(arrays, expected)):
                        actual = cp.asnumpy(array)
                        require(
                            np.array_equal(np.isfinite(actual), np.isfinite(reference)),
                            f"Nonfinite pattern differs in state field {index}.",
                        )
                        if np.issubdtype(actual.dtype, np.inexact):
                            mask = np.isfinite(reference)
                            finite_values += int(mask.sum())
                            if mask.any():
                                max_error = max(
                                    max_error,
                                    float(np.max(np.abs(actual[mask] - reference[mask]))),
                                )
                            require(
                                np.allclose(
                                    actual, reference, rtol=1e-11, atol=1e-12, equal_nan=True
                                ),
                                f"GPU numerical validation failed in state field {index}.",
                            )
                        else:
                            require(
                                np.array_equal(actual, reference),
                                f"Integer state field {index} differs.",
                            )
                    require(
                        finite_values > 0,
                        "Validation contained no finite floating-point values.",
                    )
                    report["validation"] = dict(
                        status="passed",
                        fields=len(arrays),
                        finite_values=finite_values,
                        max_abs_error=max_error,
                        rtol=1e-11,
                        atol=1e-12,
                    )
                    del expected
            report["scalar_inventory_validation"] = "passed_after_both_arms"
            report["rayleigh_reset_evidence"] = [
                dict(
                    initial_value=value,
                    requested_dtime=getattr(callback, "keywords", {}).get("dtime"),
                )
                for _, name, value in scalar_state
                if name == "_dtime_previous_substep"
            ]

            seed = case()["order_seed"]
            schedule = timing_schedule(
                case()["quartets"],
                seed,
                placebo=True,
            )
            report["timing_design"] = dict(seed=seed, schedule=schedule, interleaved=True)
            phase_data = {"intervention": report["blocks"], "placebo": []}
            for phase, quartet, order in schedule:
                print(
                    f"Fusion {case()['name']} {phase} quartet {quartet + 1}/{case()['quartets']} {order}",
                    flush=True,
                )
                phase_blocks = phase_data[phase]
                for arm in order:
                    STATE["arm"] = "A" if phase == "placebo" else arm
                    restore()
                    for _ in range(warmup):
                        invoke()
                    samples = []
                    for _ in range(rounds):
                        before = metric_offsets()
                        start = time.perf_counter_ns()
                        invoke()
                        wall_ms = (time.perf_counter_ns() - start) / 1e6
                        programs = device_samples(before)
                        samples.append(
                            dict(
                                device_ms=sum(programs.values()),
                                wall_ms=wall_ms,
                                programs=programs,
                            )
                        )
                    names = set(samples[0]["programs"])
                    require(
                        all(set(s["programs"]) == names for s in samples),
                        "Hot-loop program coverage changed.",
                    )
                    phase_blocks.append(
                        dict(
                            arm=arm,
                            quartet=quartet,
                            order=order,
                            samples=samples,
                            device_ms=sum(
                                statistics.median(s["programs"][n] for s in samples) for n in names
                            ),
                            wall_ms=statistics.median(s["wall_ms"] for s in samples),
                            programs={
                                n: statistics.median(s["programs"][n] for s in samples)
                                for n in names
                            },
                        )
                    )
            if phase_data["placebo"]:
                report["placebo"] = dict(
                    blocks=phase_data["placebo"], paired=paired_summary(phase_data["placebo"])
                )
            report["paired"] = paired_summary(report["blocks"])
            verify_scalar_inventory(roots, scalar_state)
            report["options"] = STATE["options"]
            report["generated"] = STATE["audits"]
            require(report["grid_dimensions"]["levels"] == 120, "Expected 120 vertical levels.")
            require(
                len(arrays) == 148,
                "State field inventory differs from the measured 148 fields; review before comparing.",
            )
            for target in TARGETS:
                require(
                    any(a["program"] == target for a in STATE["audits"]),
                    f"No fresh generated-code evidence for {target}.",
                )
            require(
                all(
                    set(b["programs"]) == set(report["blocks"][0]["programs"])
                    for b in report["blocks"] + report["placebo"]["blocks"]
                ),
                "Program coverage changed between arms/blocks.",
            )
            report["status"] = "complete"
            self.request.config._fusion_completed = True
            save_json(Path(output).with_name("analysis.json"), analyze(report))
        except BaseException as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            self.has_error = True
            raise
        finally:
            save_json(output, report)
            self.extra_info["causal_report"] = output


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.argname == "benchmark" and request.node.nodeid.endswith(TARGET):
        fixture = FusionBenchmark(request)
        fixturedef.cached_result = (fixture, fixturedef.cache_key(request), None)
        return fixture
