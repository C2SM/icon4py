# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import typing

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import pytest
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common import (
    backend_configuration as backend_cfg,
    dimension,
    field_type_aliases as fa,
    model_backends,
    model_options,
)
from icon4py.model.common.model_options import customize_backend, setup_program


@pytest.fixture(autouse=True)
def clear_backend_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_SIZE", raising=False)


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


def test_custom_backend_with_external_workspace_config_and_no_explicit_device() -> None:
    backend = customize_backend(
        None,
        {"backend_factory": model_backends.make_custom_dace_backend, "device": None},
        backend_config=backend_cfg.BackendConfig(workspace_size=4096),
    )
    assert backend is not None
    assert hasattr(backend, "external_workspace")
    assert backend.external_workspace[gtx.DeviceType.CPU].size == 4096


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


def test_customize_backend_applies_dace_options_per_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, typing.Any]] = []
    set_program_options = model_options._set_program_specific_dace_options
    set_device_options = model_options._set_device_specific_dace_options

    def spy_program_options(program_name: str, *args: typing.Any) -> None:
        calls.append(("program", program_name))
        set_program_options(program_name, *args)

    def spy_device_options(device: model_backends.DeviceType | None, **kwargs: typing.Any) -> None:
        calls.append(("device", device))
        set_device_options(device, **kwargs)

    monkeypatch.setattr(model_options, "_set_program_specific_dace_options", spy_program_options)
    monkeypatch.setattr(model_options, "_set_device_specific_dace_options", spy_device_options)

    def vertically_implicit_solver_at_predictor_step() -> None: ...

    descriptor = {
        "backend_factory": model_backends.make_custom_dace_backend,
        "device": model_backends.CPU,
    }
    backend = model_options.with_backend(program_return_field, descriptor).backend
    assert isinstance(backend, gtx_backend.Backend)
    customize_backend(vertically_implicit_solver_at_predictor_step, descriptor)
    assert calls == [
        ("program", "program_return_field"),
        ("device", model_backends.CPU),
        ("program", "vertically_implicit_solver_at_predictor_step"),
        ("device", model_backends.CPU),
    ]
    assert descriptor == {
        "backend_factory": model_backends.make_custom_dace_backend,
        "device": model_backends.CPU,
    }


def test_backend_config_travels_with_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = {"backend_factory": model_backends.make_custom_dace_backend, "device": None}
    descriptor_config = backend_cfg.BackendConfig(workspace_size=4096)
    backend_like = model_options.with_backend_config(descriptor, descriptor_config)
    assert "backend_config" not in descriptor

    class _GetOptionsCalled(Exception): ...

    received: list[tuple[backend_cfg.BackendConfig | None, dict]] = []

    def spy_get_options(program_name: str, *, backend_config, **backend_descriptor):  # type: ignore[no-untyped-def]
        received.append((backend_config, backend_descriptor))
        raise _GetOptionsCalled

    monkeypatch.setattr(model_options, "get_options", spy_get_options)
    with pytest.raises(_GetOptionsCalled):
        customize_backend(program_return_field, backend_like)
    # an explicit argument takes precedence over the one in the descriptor
    explicit_config = backend_cfg.BackendConfig(workspace_size=8192)
    with pytest.raises(_GetOptionsCalled):
        customize_backend(program_return_field, backend_like, backend_config=explicit_config)

    assert [config for config, _ in received] == [descriptor_config, explicit_config]
    assert all("backend_config" not in options for _, options in received)


def test_with_backend_config_keeps_concrete_backend() -> None:
    backend = model_backends.make_custom_dace_backend(device=model_backends.CPU)
    config = backend_cfg.BackendConfig(workspace_size=4096)
    assert model_options.with_backend_config(backend, config) is backend
    assert model_options.with_backend_config(None, config) is None
    assert model_options.with_backend_config(model_backends.CPU, config) == {
        "device": model_backends.CPU,
        "backend_config": config,
    }


def test_rocm_device_options_are_applied_with_program_options() -> None:
    options = model_options.get_options(
        "vertically_implicit_solver_at_predictor_step",
        backend_config=None,
        device=model_backends.DeviceType.ROCM,
    )["optimization_args"]
    assert options["fuse_scan_inputs"] is True
    assert options["gpu_block_size_2d"] == (256, 1, 1)
    assert options["blocking_size"] == 4
    assert options["transient_memory_mode"] == gtx_transformations.TransientMemoryMode.PERSISTENT


def test_customize_backend_keeps_foreign_concrete_backend() -> None:
    backend = model_backends.make_custom_dace_backend(device=model_backends.CPU)
    assert customize_backend(program_return_field, backend) is backend


@pytest.mark.parametrize(
    "program, expected",
    [
        (
            "compute_rho_theta_pgrad_and_update_vn",
            {
                "vertical_split_allow_shared_data": True,
                "vertical_split_shared_data_dimension": dimension.KDim,
            },
        ),
        (
            "vertically_implicit_solver_at_predictor_step",
            {"fuse_scan_inputs": True, "scan_fusion_scope": "field_operator"},
        ),
        (
            "vertically_implicit_solver_at_corrector_step",
            {"fuse_scan_inputs": True, "scan_fusion_scope": "field_operator"},
        ),
        ("another_program", {}),
    ],
)
def test_selected_fusion_options(program: str, expected: dict) -> None:
    descriptor = {"optimization_args": {"optimization_hooks": {}}}
    options = model_options.get_dace_options(program, None, **descriptor)["optimization_args"]
    assert all(options[key] == value for key, value in expected.items())
    assert "vertical_split_shared_data" not in options
    assert options.get("fuse_scan_inputs", False) == expected.get("fuse_scan_inputs", False)
    assert options.get("vertical_split_allow_shared_data", False) == expected.get(
        "vertical_split_allow_shared_data", False
    )
    assert descriptor == {"optimization_args": {"optimization_hooks": {}}}


@pytest.mark.parametrize(
    "program, flag",
    [
        ("compute_rho_theta_pgrad_and_update_vn", "vertical_split_allow_shared_data"),
        ("vertically_implicit_solver_at_predictor_step", "fuse_scan_inputs"),
        ("vertically_implicit_solver_at_corrector_step", "fuse_scan_inputs"),
    ],
)
def test_fusion_can_be_disabled(program: str, flag: str) -> None:
    options = model_options.get_dace_options(program, None, optimization_args={flag: False})
    assert options["optimization_args"][flag] is False
