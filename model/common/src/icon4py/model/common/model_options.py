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
from gt4py._core.definitions import (
    Scalar,
    is_scalar_type,  # TODO(havogt): Should this function be public API?
)
from gt4py.next import Field
from gt4py.next.common import OffsetProvider
from gt4py.next.ffront.decorator import Program

from icon4py.model.common.model_backends import make_custom_dace_backend, make_custom_gtfn_backend


def dict_values_to_list(d: dict[str, typing.Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def get_options(program_name: str, arch: str, **backend):
    backend_kind = backend.get("backend_kind", "default")
    return dict(backend_kind=backend_kind)


def customize_backend(program: str = "", arch: str = "", **backend):
    program_id = str(program.past_stage.past_node.id) if program != "" else program
    options = get_options(program_id, arch, **backend)
    if options["backend_kind"] == "dace":
        backend_func = make_custom_dace_backend
    elif options["backend_kind"] == "gtfn":
        backend_func = make_custom_gtfn_backend
    on_gpu = backend["device"] == "gpu"
    custom_backend = backend_func(
        on_gpu=on_gpu,
        **options,
    )
    return custom_backend


def setup_program(
    program: Program,
    backend: gtx.backend.Backend
    | typing.Literal["gpu", "cpu"]
    | dict[str, typing.Any]
    | None = None,
    constant_args: dict[str, Field | Scalar] | None = None,
    variants: dict[str, list[Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: OffsetProvider | None = None,
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

    if isinstance(backend, gtx.backend.Backend):
        custom_backend = backend
    else:
        # customized backend params
        custom_backend = customize_backend(**backend)

    bound_static_args = {k: v for k, v in constant_args.items() if is_scalar_type(v)}
    static_args_program = program.with_backend(custom_backend).compile(
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
        **variants,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
