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
from types import MappingProxyType
from typing import Any

import cffi
import numpy as np
from functional.common import Dimension, DimensionKind
from functional.iterator.embedded import np_as_located_field

from icon4py.common.dimension import KDim, VertexDim
from icon4py.py2f.typing_utils import parse_annotation


FFI_DEF_EXTERN_DECORATOR = "@ffi.def_extern()"

CFFI_GEN_DECORATOR = "@CffiMethod.register"


class CffiMethod:
    _registry = {}

    @classmethod
    def register(cls, func):
        try:
            cls._registry[func.__module__].append(func.__name__)
        except KeyError:
            cls._registry[func.__module__] = [func.__name__]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]


def with_cffi_gen(func):
    @functools.wraps(func)
    def _cffi_gen(*args, **kwargs):
        return func(*args, **kwargs)

    return _cffi_gen


def generate_and_compile_cffi_plugin(
    plugin_name: str, c_header: str, module_name: str, build_path="."
):
    """
    Create C shared library.

    Create a linkable C library and F90 interface for the functions in the python module
    {module_name} that are decorated with '@CffiMethod.register'.

    Args:
        plugin_name: name of the plugin, a linkable C library with the name
            'lib{plugin_name}.so' will be created in the {build_path} folder'
        c_header: C type header signature for the python functions.
        module_name:  python module name that contains python functions corresponding
            to the signature in the '{c_header}' string, these functions must be decorated
            with @CffiMethod.register and the file must contain the import
        build_path: *optional* path to build directory

    """
    python_src_file = f"{module_name.split('.')[-1]}.py"
    c_header_file = plugin_name + ".h"
    with open("/".join([build_path, c_header_file]), "w") as f:
        f.write(c_header)

    builder = cffi.FFI()

    builder.embedding_api(c_header)
    builder.set_source(plugin_name, f'#include "{c_header_file}"')

    with open(python_src_file) as f:
        module = f.read()

    module = f"from {plugin_name} import ffi\n{module}".replace(
        CFFI_GEN_DECORATOR, FFI_DEF_EXTERN_DECORATOR
    )

    builder.embedding_init_code(module)
    builder.compile(tmpdir=build_path, target=f"lib{plugin_name}.*", verbose=True)


def to_fields(
    dim_sizes: dict[Dimension, int]
):  # pass lengths of dimensions to the decorator??
    ffi = cffi.FFI()
    dim_sizes = dim_sizes

    def _dim_sizes(dims: list[Dimension]) -> tuple[int, int]:
        v_size = None
        h_size = None
        for d in dims:
            if d.kind == DimensionKind.VERTICAL:
                v_size = dim_sizes[d]
            elif d.kind == DimensionKind.HORIZONTAL:
                h_size = dim_sizes[d]
        return h_size, v_size

    def unpack(ptr, size_x, size_y) -> np.ndarray:
        """
        unpacks a 2d c/fortran field into a numpy array.

        :param ptr: c_pointer to the field
        :param size_x: col size (since its called from fortran)
        :param size_y: row size
        :return: a numpy array with shape=(size_y, size_x)
        and dtype = ctype of the pointer
        """
        # for now only 2d, invert for row/column precedence...
        shape = (size_y, size_x)
        length = np.prod(shape)
        c_type = ffi.getctype(ffi.typeof(ptr).item)
        ar = np.frombuffer(
            ffi.buffer(ptr, length * ffi.sizeof(c_type)),
            dtype=np.dtype(c_type),
            count=-1,
            offset=0,
        ).reshape(shape)
        return ar

    def _to_fields_decorator(func):
        @functools.wraps(func)
        def wrapper(
            *args, **kwargs
        ):  # these are the args of the decorated function ie to_fields(func(*args, **kwargs))
            signature = inspect.signature(func)
            print(
                f"sizes passed to decorator horizontal = {dim_sizes[VertexDim]}, vertical ={dim_sizes[KDim]}"
            )
            print(f"signature of func {signature}")
            parameters: MappingProxyType[str, inspect.Parameter] = signature.parameters
            print(f"parameters = {parameters}")
            bind: inspect.BoundArguments = signature.bind(*args, **kwargs)

            arguments: OrderedDict[str, Any] = bind.arguments
            print(f"arguments = {arguments}")
            f_args = []
            for name, argument in arguments.items():
                # TODO: scalar types, tests, simplify this.
                print(f"name={name} argument={argument}")
                annotation = parameters[name].annotation
                dims, dtype = parse_annotation(annotation)
                print(
                    f" type hint of {name} is {annotation}: dims = {dims}, dtype = {dtype}"
                )
                (size_h, size_v) = _dim_sizes(dims)
                ar = unpack(argument, size_h, size_v)
                field = np_as_located_field(dims)(ar)
                f_args.append(field)
            return func(*f_args, **kwargs)

        return wrapper

    return _to_fields_decorator
