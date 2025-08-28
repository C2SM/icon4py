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
    is_scalar_type,  # TODO(havogt): Should this function be public API?
)
from gt4py.next import backend


def dict_values_to_list(d: dict[str, typing.Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def setup_program(
    backend: backend.Backend,
    program: gtx.program,
    constant_args: dict | None = None,
    variants: dict | None = None,
    horizontal_sizes: dict | None = None,
    vertical_sizes: dict | None = None,
    offset_provider: dict | None = None,
) -> typing.Callable[..., None]:
    """
    This function processes pre-compiled args and feeds some to the gt4py `compile` function.
    Args:
        - backend: pre-set backend at run time,
        - program_func: gt4py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal scalars,
        - vertical_sizes: vertical scalars,
        - offset_provider: gt4py offset_provider,
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    bound_static_args = {k: v for k, v in constant_args.items() if is_scalar_type(v)}
    static_args_program = program.with_backend(backend).compile(
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
