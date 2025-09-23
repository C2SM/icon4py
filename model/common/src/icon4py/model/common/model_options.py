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

from icon4py.model.common import model_backends
from icon4py.model.common.utils.device_utils import is_BackendDescriptor


def dict_values_to_list(d: dict[str, typing.Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def customize_backend(
    backend: model_backends.DeviceType | model_backends.BackendDescriptor,
) -> gtx_typing.Backend:
    if isinstance(backend, model_backends.DeviceType):
        backend = {"device": backend}
    # TODO(havogt): implement the lookup function as below
    # options = get_options(program_name, arch, **backend) # noqa: ERA001
    backend_func = backend.get("backend_factory", model_backends.make_custom_gtfn_backend)
    device = backend.get("device", model_backends.DeviceType.CPU)
    custom_backend = backend_func(
        device=device,
    )
    return custom_backend


def setup_program(
    program: gtx_typing.Program,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
) -> typing.Callable[..., None]:
    """
    This function processes arguments to the GT4Py program. It
    - binds arguments that don't change during model run ('constant_args', 'horizontal_sizes', "vertical_sizes');
    - inlines scalar arguments into the GT4Py program at compile-time (via GT4Py's 'compile').
    Args:
        - backend: GT4Py backend,
        - program: GT4Py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal domain bounds,
        - vertical_sizes: vertical domain bounds,
        - offset_provider: GT4Py offset_provider,
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    if isinstance(backend, gtx.DeviceType) or is_backend_descriptor(backend):
        backend = customize_backend(backend)

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            enable_jit=False,
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
