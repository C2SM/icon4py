# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import functools
import inspect
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

import cffi
import gt4py.next as gtx
import numpy as np
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next.type_system.type_translation import from_type_hint

from icon4pytools.py2fgen.common import parse_type_spec


PROGRAM_DECORATOR = "@program"


class CffiMethod:
    _registry: dict[str, list[str]] = {}

    @classmethod
    def register(cls, func):
        cls._registry.setdefault(func.__module__, []).append(func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]


def generate_and_compile_cffi_plugin(
    plugin_name: str, c_header: str, python_wrapper: str, build_path: Path = Path(".")
):
    """
    Create C shared library.

    Create a linkable C library and F90 interface for the functions in the python module
    {module_name} that are decorated with '@CffiMethod.register'.

    Args:
        plugin_name: name of the plugin, a linkable C library with the name
            'lib{plugin_name}.so' will be created in the {build_path} folder'
        c_header: C type header signature for the python functions.
        python_wrapper: Python code wrapping the original function to be exposed.
        build_path: *optional* path to build directory

    """
    c_header_file = plugin_name + ".h"
    header_file_path = build_path / c_header_file

    with open(header_file_path, "w") as f:
        f.write(c_header)

    builder = cffi.FFI()

    builder.embedding_api(c_header)
    builder.set_source(plugin_name, f'#include "{c_header_file}"')
    builder.embedding_init_code(python_wrapper)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}.*", verbose=True)  # type: ignore


class UnknownDimensionException(Exception):
    """Raised if a Dimension is unknown to the interface generation."""

    pass


def to_fields(dim_sizes: dict[Dimension, int]):
    """
    Pack/Unpack Fortran 2d arrays to numpy arrays with using CFFI frombuffer.

    Args:
        dim_sizes: dictionary containing the sizes of the dimension.

    #TODO (magdalena)  handle dimension sizes in a better way?
    """
    ffi = cffi.FFI()
    dim_sizes = dim_sizes

    def _dim_sizes(dims: Sequence[Dimension]) -> tuple[int | None, int | None]:
        """Extract the size of dimension from a dictionary."""
        v_size = None
        h_size = None
        for d in dims:
            print(d)
            if d not in dim_sizes.keys():
                raise UnknownDimensionException(
                    f"size of dimension '{d}' not defined in '{dim_sizes.keys()}'"
                )
            if d.kind == DimensionKind.VERTICAL or d.kind == DimensionKind.LOCAL:
                v_size = dim_sizes[d]
            elif d.kind == DimensionKind.HORIZONTAL:
                h_size = dim_sizes[d]
        return h_size, v_size

    def _unpack(ptr, size_h, size_v, dtype) -> np.ndarray:
        """
        Unpack a 2d c/fortran field into a numpy array.

        :param dtype: expected type of the fields
        :param ptr: c_pointer to the field
        :param size_h: length of horizontal dimension
        :param size_v: length of vertical dimension
        :return: a numpy array with shape=(size_h, size_v)
        and dtype = ctype of the pointer
        """
        shape = (size_h, size_v)
        length = np.prod(shape)
        c_type = ffi.getctype(ffi.typeof(ptr).item)
        # TODO (magdalena) fix dtype handling use SCALARTYPE?
        mem_size = np.dtype(c_type).itemsize
        ar = np.frombuffer(  # type: ignore[call-overload]
            ffi.buffer(ptr, length * mem_size),
            dtype=np.dtype(c_type),
            count=-1,
            offset=0,
        ).reshape(shape)
        return ar

    def _to_fields_decorator(func):
        @functools.wraps(func)
        def _wrapper(
            *args, **kwargs
        ):  # these are the args of the decorated function ie to_fields(func(*args, **kwargs))
            signature = inspect.signature(func)
            parameters: MappingProxyType[str, inspect.Parameter] = signature.parameters
            arguments: OrderedDict[str, Any] = signature.bind(*args, **kwargs).arguments
            f_args = []
            for name, argument in arguments.items():
                ar = _transform_arg(argument, name, parameters)
                f_args.append(ar)
            return func(*f_args, **kwargs)

        def _transform_arg(argument, name, parameters):
            type_spec = from_type_hint(parameters[name].annotation)
            dims, dtype = parse_type_spec(type_spec)
            if dims:
                (size_h, size_v) = _dim_sizes(dims)
                ar = _unpack(argument, size_h, size_v, dtype)
                ar = gtx.as_field(domain=dims, data=ar, dtype=ar.dtype)
            else:
                ar = argument
            return ar

        return _wrapper

    return _to_fields_decorator
