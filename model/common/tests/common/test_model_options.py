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

from icon4py.model.common import (
    backend_configuration as backend_cfg,
    field_type_aliases as fa,
    model_backends,
)
from icon4py.model.common.model_options import customize_backend, setup_program


@pytest.fixture(autouse=True)
def clear_backend_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_SIZE", raising=False)
    monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_ALIGNMENT", raising=False)


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
        backend_config=backend_cfg.BackendConfig(
            workspace_size=4096,
            workspace_alignment=256,
        ),
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
