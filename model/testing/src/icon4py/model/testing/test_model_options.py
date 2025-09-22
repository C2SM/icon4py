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

from icon4py.model.atmosphere.diffusion.diffusion_utils import scale_k
from icon4py.model.common import field_type_aliases as fa

from icon4py.model.common import model_backends
from icon4py.model.common.model_options import customize_backend, setup_program


@gtx.field_operator
def field_op_return_field(field: fa.KField[float], factor: float) -> fa.KField[float]:
    return field + factor

@gtx.program
def program_return_field(field: fa.KField[float], factor: float, scaled_field: fa.KField[float]):
    field_op_return_field(field, factor, out=scaled_field)

@pytest.mark.parametrize(
    "backend_factory, expected_backend",
    [
        (model_backends.make_custom_gtfn_backend, "gtfn"),
        (model_backends.make_custom_dace_backend, "dace"),
    ],
)
def test_custom_backend_options(backend_factory: typing.Callable, expected_backend: str) -> None:
    backend_options: dict = {
        "backend_factory": backend_factory,
        "device": model_backends.CPU,
    }
    backend = customize_backend(backend_options)
    backend_name = expected_backend + "_cpu"
    # TODO: test should be improved to work without string comparison
    assert repr(model_backends.BACKENDS[backend_name]) == repr(backend)


def test_custom_backend_device() -> None:
    device = model_backends.CPU
    backend = customize_backend(device)
    default_backend = "gtfn_cpu"
    # TODO: test should be improved to work without string comparison
    assert repr(model_backends.BACKENDS[default_backend]) == repr(backend)


@pytest.mark.parametrize(
    "backend",
    [
        model_backends.BACKENDS["gtfn_cpu"],
        model_backends.CPU,
        {"backend_factory": model_backends.make_custom_gtfn_backend, "device": model_backends.CPU},
        {"backend_factory": model_backends.make_custom_gtfn_backend},
        {"device": model_backends.CPU},
    ],
)
def test_setup_program_defaults(
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescription
    | None,
) -> None:
    partial_program = setup_program(backend=backend, program=scale_k)
    backend = model_backends.BACKENDS["gtfn_cpu"]
    expected_partial = functools.partial(
        scale_k.with_backend(backend).compile(
            enable_jit=False,
            offset_provider={},
        ),
        offset_provider={},
    )
    # TODO: test should be improved to work without string comparison
    assert repr(partial_program) == repr(expected_partial)


@pytest.mark.parametrize(
    "backend_params, expected_backend",
    [
        (model_backends.BACKENDS["gtfn_gpu"], "gtfn_gpu"),
        (model_backends.BACKENDS["embedded"], "embedded"),
        (
            {
                "backend_factory": model_backends.make_custom_dace_backend,
                "device": model_backends.GPU,
            },
            "dace_gpu",
        ),
        ({"backend_factory": model_backends.make_custom_dace_backend}, "dace_cpu"),
        ({"device": model_backends.GPU}, "gtfn_gpu"),
    ],
)
def test_setup_program_specify_inputs(
    backend_params: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescription
    | None,
    expected_backend: str,
) -> None:
    partial_program = setup_program(backend=backend_params, program=scale_k)
    backend = model_backends.BACKENDS[expected_backend]
    expected_partial = functools.partial(
        scale_k.with_backend(backend).compile(
            enable_jit=False,
            offset_provider={},
        ),
        offset_provider={},
    )
    # TODO: test should be improved to work without string comparison
    assert repr(partial_program) == repr(expected_partial)
