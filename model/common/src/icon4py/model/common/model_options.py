# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import typing

from gt4py._core.definitions import is_scalar_type
from gt4py.eve.utils import FrozenNamespace
from gt4py.next import backend


class RayleighType(FrozenNamespace[int]):
    #: classical Rayleigh damping, which makes use of a reference state.
    CLASSIC = 1
    #: Klemp (2008) type Rayleigh damping
    KLEMP = 2


def dict_values_to_list(d: dict[str, typing.Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


# flake8: noqa: B006
def program_compile_time(
    backend: backend.Backend,
    program_func: typing.Callable,
    constant_args: dict = {},
    variants: dict = {},
    horizontal_sizes: dict = {},
    vertical_sizes: dict = {},
    offset_provider: dict = {},
) -> typing.Callable[..., None]:
    """
    backend: pre-set backend at run time,
    program_func: gt4py program,
    constant_args: constant fields and scalars,
    variants: list of all scalars potential values from which one is selected at run time,
    horizontal_sizes: horizontal scalars,
    vertical_sizes: vertical scalars,
    offset_provider: gt4py offset_provider,
    """
    bound_static_args = {k: v for k, v in constant_args.items() if is_scalar_type(v)}
    static_args_program = program_func.with_backend(backend).compile(
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
