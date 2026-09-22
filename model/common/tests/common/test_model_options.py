# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import sys
import typing
from types import SimpleNamespace

import dace
import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import pytest
from gt4py.next.program_processors.runners import dace as dace_backend

from icon4py.model.common import field_type_aliases as fa, model_backends, model_options
from icon4py.model.common.model_options import customize_backend, setup_program


@gtx.field_operator  # type: ignore[call-overload]
def field_op_return_field(field: fa.CellKField[float], factor: float) -> fa.CellKField[float]:
    return field + factor


@gtx.program  # type: ignore[call-overload]
def program_return_field(field: fa.CellKField[float], factor: float):  # type: ignore[no-untyped-def]
    field_op_return_field(field, factor, out=field)


@pytest.mark.parametrize(
    "backend_factory, expected_backend",
    [
        (
            model_backends.make_custom_gtfn_backend,
            model_backends.make_custom_gtfn_backend(device=model_backends.CPU),
        ),
        (
            model_backends.make_custom_dace_backend,
            model_backends.make_custom_dace_backend(device=model_backends.CPU),
        ),
    ],
)
def test_custom_backend_options(backend_factory: typing.Callable, expected_backend: str) -> None:
    backend_options: dict = {
        "backend_factory": backend_factory,
        "device": model_backends.CPU,
    }
    backend = customize_backend(None, backend_options)
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(expected_backend) == repr(backend)


def test_custom_backend_device() -> None:
    device = model_backends.CPU
    backend = customize_backend(None, device)
    default_backend = model_backends.make_custom_dace_backend(device=device)
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(default_backend) == repr(backend)


@pytest.mark.parametrize(
    "backend",
    [
        model_backends.make_custom_dace_backend(device=model_backends.CPU),  # conrete backend
        model_backends.CPU,
        {"backend_factory": model_backends.make_custom_dace_backend, "device": model_backends.CPU},
        {"backend_factory": model_backends.make_custom_dace_backend},
        {"device": model_backends.CPU},
    ],
)
def test_setup_program_defaults(
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
) -> None:
    testee = setup_program(backend=backend, program=program_return_field)
    expected_backend = model_backends.make_custom_dace_backend(device=model_backends.CPU)
    expected_program = functools.partial(
        program_return_field.with_backend(expected_backend)
        .with_compilation_options(enable_jit=False)
        .compile(
            offset_provider={},
        ),
        offset_provider={},
    )
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(testee) == repr(expected_program)


@pytest.mark.parametrize(
    "backend_params, expected_backend",
    [
        (model_backends.BACKENDS["embedded"], model_backends.BACKENDS["embedded"]),
        (
            {
                "backend_factory": model_backends.make_custom_dace_backend,
                "device": model_backends.GPU,
            },
            model_backends.make_custom_dace_backend(device=model_backends.GPU),
        ),
        (
            {"backend_factory": model_backends.make_custom_dace_backend},
            model_backends.make_custom_dace_backend(device=model_backends.CPU),
        ),
        (
            {"device": model_backends.GPU},
            model_backends.make_custom_dace_backend(device=model_backends.GPU),
        ),
    ],
)
def test_setup_program_specify_inputs(
    backend_params: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    expected_backend: gtx_typing.Backend | None,
) -> None:
    testee = setup_program(backend=backend_params, program=program_return_field)
    if expected_backend is None:
        expected_program = functools.partial(
            program_return_field.with_backend(expected_backend), offset_provider={}
        )
    else:
        expected_program = functools.partial(
            program_return_field.with_backend(expected_backend)
            .with_compilation_options(enable_jit=False)
            .compile(
                offset_provider={},
            ),
            offset_provider={},
        )
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(testee) == repr(expected_program)


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_theta_fusion_options(monkeypatch, setting):
    monkeypatch.delenv("ICON4PY_DACE_THETA_FUSION", raising=False)
    if setting is not None:
        monkeypatch.setenv("ICON4PY_DACE_THETA_FUSION", setting)
    hooks = model_options.gtx_transformations.GT4PyAutoOptHook
    descriptor = {"optimization_args": {"optimization_hooks": {}}}
    options = model_options.get_dace_options(
        "compute_rho_theta_pgrad_and_update_vn", None, **descriptor
    )
    registered = options["optimization_args"]["optimization_hooks"]
    assert (hooks.TopLevelDataFlowVerticalSplitCallBack in registered) == (setting == "1")
    assert descriptor == {"optimization_args": {"optimization_hooks": {}}}
    other = model_options.get_dace_options("another_program", None, **descriptor)
    assert (
        hooks.TopLevelDataFlowVerticalSplitCallBack
        not in other["optimization_args"]["optimization_hooks"]
    )


def test_theta_fusion_configuration_errors(monkeypatch):
    program = "compute_rho_theta_pgrad_and_update_vn"
    monkeypatch.setenv("ICON4PY_DACE_THETA_FUSION", "yes")
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        model_options.get_dace_options(program, None)
    monkeypatch.setenv("ICON4PY_DACE_THETA_FUSION", "1")
    key = model_options.gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowVerticalSplitCallBack
    with pytest.raises(ValueError, match="existing vertical-split callback"):
        model_options.get_dace_options(
            program, None, optimization_args={"optimization_hooks": {key: lambda *args: True}}
        )
    monkeypatch.setattr(
        model_options.map_fusion_extended.VerticalSplitMapRange, "__properties__", {}
    )
    with pytest.raises(RuntimeError, match="requires the GT4Py"):
        model_options.get_dace_options(program, None)


def test_theta_fusion_callback_resets_and_restricts_candidates():
    edge, level = "i_Edge_gtx_horizontal", "i_K_gtx_vertical"
    first = dace.nodes.Map("producer", [edge, level], dace.subsets.Range.from_string("1:5,0:12"))
    second = dace.nodes.Map("consumer", [edge, level], dace.subsets.Range.from_string("1:5,0:3"))
    transformation = SimpleNamespace(
        access_node=SimpleNamespace(data="theta_v_at_edges_on_model_levels"),
        allow_shared_data=False,
    )
    callback = model_options._dace_select_theta_split
    assert callback(transformation, first, second, None, None)
    assert transformation.allow_shared_data
    transformation.access_node.data = "another_field"
    assert callback(transformation, first, second, None, None)
    assert not transformation.allow_shared_data
    transformation.access_node.data = "theta_v_at_edges_on_model_levels"
    second.range = dace.subsets.Range.from_string("2:5,0:3")
    assert callback(transformation, first, second, None, None)
    assert not transformation.allow_shared_data
    assert callback(transformation, first, first, None, None)
    assert not transformation.allow_shared_data


@pytest.mark.parametrize("setting, expected_maps", [("0", 3), ("1", 2)])
def test_theta_fusion_through_model_options_and_auto_optimizer(monkeypatch, setting, expected_maps):
    monkeypatch.setenv("ICON4PY_DACE_THETA_FUSION", setting)
    theta = "theta_v_at_edges_on_model_levels"
    edge, level = "i_Edge_gtx_horizontal", "i_K_gtx_vertical"
    sdfg = dace.SDFG("model_options_theta_fusion")
    for name in ("input", theta, "rho", "wind"):
        sdfg.add_array(name, [6, 12], dace.float64)
    state = sdfg.add_state()
    access = {name: state.add_access(name) for name in sdfg.arrays}
    state.add_mapped_tasklet(
        "producer",
        {edge: "1:5", level: "0:12"},
        {"a": dace.Memlet(f"input[{edge},{level}]")},
        "t = a + 1; r = a * 2",
        {"t": dace.Memlet(f"{theta}[{edge},{level}]"), "r": dace.Memlet(f"rho[{edge},{level}]")},
        input_nodes={access["input"]},
        output_nodes={access[theta], access["rho"]},
        external_edges=True,
    )
    for name, band in (("lower", "0:3"), ("upper", "3:12")):
        state.add_mapped_tasklet(
            name,
            {edge: "1:5", level: band},
            {"t": dace.Memlet(f"{theta}[{edge},{level}]")},
            "w = t * 3",
            {"w": dace.Memlet(f"wind[{edge},{level}]")},
            input_nodes={access[theta]},
            output_nodes={access["wind"]},
            external_edges=True,
        )
    options = model_options.get_dace_options("compute_rho_theta_pgrad_and_update_vn", None)
    model_options.gtx_transformations.gt_auto_optimize(
        sdfg, gpu=False, validate=True, validate_all=True, **options.get("optimization_args", {})
    )
    assert sum(isinstance(node, dace.nodes.MapEntry) for node in state.nodes()) == expected_maps
    assert not sdfg.arrays[theta].transient
    assert not sdfg.arrays["rho"].transient


@pytest.mark.parametrize("setting", [None, "0", "1"])
@pytest.mark.parametrize(
    "program",
    [
        "vertically_implicit_solver_at_predictor_step",
        "vertically_implicit_solver_at_corrector_step",
    ],
)
def test_solver_fusion_options(monkeypatch, setting, program):
    monkeypatch.delenv("ICON4PY_DACE_SOLVER_FUSION", raising=False)
    if setting is not None:
        monkeypatch.setenv("ICON4PY_DACE_SOLVER_FUSION", setting)
    options = model_options.get_dace_options(program, None)
    assert options["optimization_args"].get("fuse_scan_inputs", False) == (setting == "1")
    assert options["optimization_args"].get("scan_fusion_scope") == (
        "field_operator" if setting == "1" else None
    )
    other = model_options.get_dace_options("another_program", None)
    assert "scan_fusion_scope" not in other.get("optimization_args", {})
    assert "fuse_scan_inputs" not in other.get("optimization_args", {})


def test_solver_fusion_configuration_errors(monkeypatch):
    program = "vertically_implicit_solver_at_predictor_step"
    monkeypatch.setenv("ICON4PY_DACE_SOLVER_FUSION", "yes")
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        model_options.get_dace_options(program, None)
    monkeypatch.setenv("ICON4PY_DACE_SOLVER_FUSION", "1")
    monkeypatch.delattr(dace_backend, "scan_fusion", raising=False)
    monkeypatch.setitem(sys.modules, f"{dace_backend.__name__}.scan_fusion", None)
    with pytest.raises(RuntimeError, match="requires the GT4Py scan-input fusion support"):
        model_options.get_dace_options(program, None)
