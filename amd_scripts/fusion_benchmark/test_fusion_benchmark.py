# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: PLC0415
"""Regression tests for the reproduction method and normal backend dispatch."""

import json
import sys
import types

import fusion_core as core
import fusion_plugin as plugin
import numpy as np
import pytest


@pytest.fixture
def fake_gpu(monkeypatch):
    class Array(np.ndarray):
        @property
        def data(self):
            return types.SimpleNamespace(ptr=self.ctypes.data)

    cupy = types.SimpleNamespace(
        ndarray=Array,
        asarray=np.asarray,
        asnumpy=lambda a: np.asarray(a).copy(),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(
                deviceSynchronize=lambda: None, getDeviceProperties=lambda device: {}
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    return lambda values: np.asarray(values, dtype=float).view(Array)


def test_state_inventory_includes_runtime_scalars_and_aliases(fake_gpu):
    Model = type("Model", (), {"__module__": "icon4py.model.example"})
    model = Model()
    model.values = fake_gpu([1, 2, 3])
    model.alias = model.values
    model.view = model.values[1:]
    model._dtime_previous_substep = 0.0
    scalars = []
    arrays = plugin.collect_arrays([model], scalars)
    assert len(arrays) == 2  # same allocation deduplicated; distinct views retained
    assert scalars == [(model, "_dtime_previous_substep", 0.0)]
    model.new_cache = 1
    with pytest.raises(ValueError, match="inventory changed"):
        plugin.verify_scalar_inventory([model], scalars)


@pytest.mark.parametrize("quartets", [4, 6, 12])
def test_schedule_is_balanced_and_controls_matched(quartets):
    schedule = core.timing_schedule(quartets, 20260915)
    for phase in ("placebo", "intervention"):
        orders = [o for p, _, o in schedule if p == phase]
        assert orders.count("ABBA") == orders.count("BAAB") == quartets // 2
    for q in range(quartets):
        entries = [(p, o) for p, n, o in schedule if n == q]
        assert {p for p, _ in entries} == {"placebo", "intervention"}
        assert entries[0][1] == entries[1][1]


@pytest.fixture
def run_toy(tmp_path, monkeypatch, fake_gpu):
    spec = dict(
        name="combined",
        platform="amd",
        warmup=1,
        rounds=1,
        quartets=4,
        input_seed=20260910,
        order_seed=20260915,
    )
    monkeypatch.setenv("FUSION_CASE", json.dumps(spec))
    monkeypatch.setenv("FUSION_OUTPUT", str(tmp_path))
    monkeypatch.setattr(
        plugin,
        "STATE",
        dict(
            arm="A",
            build=None,
            programs={},
            patches=[],
            options=[],
            audits=[dict(program=p) for p in core.TARGETS],
        ),
    )
    Model = type("Model", (), {"__module__": "icon4py.model.example"})
    model = Model()
    model.values = [fake_gpu([float(i + 1)]) for i in range(148)]
    model._dtime_previous_substep = 0.0
    request = types.SimpleNamespace(
        node=types.SimpleNamespace(name="toy", _fusion_rng={"seed": 20260910}),
        config=types.SimpleNamespace(getoption=lambda name: "icon_benchmark_regional:120"),
        getfixturevalue=lambda name: types.SimpleNamespace(
            grid=types.SimpleNamespace(
                num_cells=1, num_edges=1, num_vertices=1, num_levels=120, limited_area=True
            )
        ),
    )
    monkeypatch.setattr(plugin, "metric_offsets", lambda: {})
    monkeypatch.setattr(
        plugin,
        "device_samples",
        lambda before: {p: (10.0 if plugin.STATE["arm"] == "A" else 9.0) for p in core.TARGETS},
    )

    def run(wrong=False, dropped_timer=False):
        calls = []

        def callback():
            calls.append(model._dtime_previous_substep)
            for field in model.values:
                field[...] += 1 + int(wrong and plugin.STATE["arm"] == "B")
            model._dtime_previous_substep = 2.0

        if dropped_timer:
            original = plugin.device_samples

            def samples(before):
                result = original(before)
                if plugin.STATE["arm"] == "B":
                    result.pop(core.SOLVERS[0])
                return result

            monkeypatch.setattr(plugin, "device_samples", samples)
        plugin.FusionBenchmark(request)(callback)
        return json.loads((tmp_path / "timing.json").read_text()), calls

    return run


def test_complete_restored_experiment(run_toy):
    report, calls = run_toy()
    assert report["validation"]["fields"] == 148
    assert report["validation"]["max_abs_error"] == 0
    assert report["restored_scalar_state"][0]["initial_value"] == 0
    assert calls[:2] == [0, 0]  # both validation arms restore the cached scalar
    assert calls[2:] == [0, 2] * 32  # restore at each block, not inside hot calls
    result = core.analyze(report)
    assert result["device"]["saving_percent"] == pytest.approx(10)
    assert result["device"]["status"] == "resolved improvement"
    assert result["device"]["placebo_mean_ms"] == 0


def test_numerical_mismatch_fails(run_toy):
    with pytest.raises(ValueError, match="numerical validation failed"):
        run_toy(wrong=True)


def test_missing_program_timer_fails(run_toy):
    with pytest.raises(ValueError, match="coverage changed"):
        run_toy(dropped_timer=True)


def test_normal_setup_selects_both_arms(monkeypatch):
    from icon4py.model.common import model_backends, model_options

    monkeypatch.setenv("FUSION_CASE", json.dumps({"name": "combined"}))
    monkeypatch.setattr(
        plugin, "STATE", dict(arm="A", build=None, programs={}, patches=[], audits=[], options=[])
    )
    captured = []
    real_options = model_options.get_dace_options

    def setup(**kwargs):
        captured.append(real_options(kwargs["program"].__name__, None, device=model_backends.CPU))
        return lambda: None

    program = types.SimpleNamespace(__name__=core.SOLVERS[0])
    pair = plugin.ProgramPair(setup, {"program": program})
    pair()
    plugin.STATE["arm"] = "B"
    pair()
    assert not captured[0]["optimization_args"].get("fuse_scan_inputs", False)
    assert captured[1]["optimization_args"]["fuse_scan_inputs"]
    assert captured[1]["optimization_args"]["scan_fusion_scope"] == "field_operator"


def test_same_name_program_compiles_distinct_variants(tmp_path, monkeypatch):
    import gt4py.next as gtx
    from dace.codegen import compiler
    from gt4py.next import config

    from icon4py.model.common import dimension as dims, model_backends, model_options

    monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_SIZE", raising=False)
    monkeypatch.setattr(config, "BUILD_JOBS", 0)
    monkeypatch.setattr(config, "BUILD_CACHE_DIR", tmp_path / "build")
    monkeypatch.setenv("FUSION_CASE", json.dumps({"name": "combined"}))
    monkeypatch.setattr(
        plugin, "STATE", dict(arm="A", build=None, programs={}, patches=[], audits=[], options=[])
    )

    @gtx.scan_operator(axis=dims.KDim, forward=True, init=(0.0, 1.0))
    def forward_scan(
        carry: tuple[gtx.float64, gtx.float64], a: gtx.float64, b: gtx.float64
    ) -> tuple[gtx.float64, gtx.float64]:
        return 0.5 * carry[0] + a, carry[1] + b

    @gtx.field_operator
    def solver(
        x: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ) -> tuple[
        gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
        gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ]:
        gamma = 2.0 * x + 1.0
        a = 3.0 * gamma
        b = gamma + 4.0
        return forward_scan(a, b)

    @gtx.program
    def vertically_implicit_solver_at_predictor_step(
        x: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
        y: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
        z: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ):
        solver(x, out=(y, z))

    builds = []
    original = compiler.generate_program_folder

    def capture(sdfg, code, *args, **kwargs):
        builds.append(
            (
                plugin.STATE["arm"],
                sdfg.build_folder,
                sum(d.transient and len(d.shape) == 2 for d in sdfg.arrays.values()),
            )
        )
        return original(sdfg, code, *args, **kwargs)

    monkeypatch.setattr(compiler, "generate_program_folder", capture)
    pair = plugin.ProgramPair(
        model_options.setup_program,
        dict(program=vertically_implicit_solver_at_predictor_step, backend=model_backends.CPU),
    )
    x = gtx.as_field((dims.CellDim, dims.KDim), np.ones((3, 12)))
    y = gtx.as_field((dims.CellDim, dims.KDim), np.zeros((3, 12)))
    z = gtx.as_field((dims.CellDim, dims.KDim), np.zeros((3, 12)))
    pair(x=x, y=y, z=z)
    expected = (y.asnumpy().copy(), z.asnumpy().copy())
    y.ndarray[:] = 0
    plugin.STATE["arm"] = "B"
    pair(x=x, y=y, z=z)
    np.testing.assert_allclose(y.asnumpy(), expected[0], rtol=1e-12)
    np.testing.assert_allclose(z.asnumpy(), expected[1], rtol=1e-12)
    assert len(builds) == 2
    assert builds[0][1] != builds[1][1]
    # CPU allocation counts need not match GPU allocation counts; distinct
    # artifact paths prove that the two translated variants were compiled.
