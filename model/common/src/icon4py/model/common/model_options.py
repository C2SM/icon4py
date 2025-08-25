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


def dict_values_to_list(d: dict[str, typing.Any]) -> dict:
    return {k: [v] for k, v in d.items()}


def program_compile_time(
    backend: backend.Backend,
    program_func: typing.Callable,
    bound_args: dict = {},  # noqa: B006
    static_args: dict = {},  # noqa: B006
    horizontal_sizes: dict = {},  # noqa: B006
    vertical_sizes: dict = {},  # noqa: B006
    offset_provider: dict = {},  # noqa: B006
):
    bound_static_args = {k: v for k, v in bound_args.items() if is_scalar_type(v)}
    static_args_program = program_func.with_backend(backend).compile(
        **dict_values_to_list(horizontal_sizes),
        **dict_values_to_list(vertical_sizes),
        **dict_values_to_list(static_args),
        **dict_values_to_list(bound_static_args),
        enable_jit=False,
        offset_provider=offset_provider,
    )
    return functools.partial(
        static_args_program,
        **bound_args,
        **static_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
