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

from enum import Enum

from gt4py.next.common import Dimension
from gt4py.next.type_system.type_specifications import FieldType, ScalarKind, ScalarType, TypeSpec


CFFI_DECORATOR = "@ffi.def_extern()"
PROGRAM_DECORATOR = "@program"


def build_array_size_args():
    array_size_args = {}
    from icon4py.model.common import dimension

    for var_name, var in vars(dimension).items():
        if isinstance(var, Dimension):
            dim_name = var_name.replace("Dim", "")
            size_name = f"n_{dim_name}"
            array_size_args[dim_name] = size_name
    return array_size_args


# TODO(samkellerhals): This should be defined as an actual function in the code so we can test it.
CFFI_UNPACK = """\
def unpack(ptr, *sizes) -> np.ndarray:
    '''
    Unpacks an n-dimensional Fortran (column-major) array into a numpy array (row-major).

    :param ptr: c_pointer to the field
    :param sizes: variable number of arguments representing the dimensions of the array in Fortran order
    :return: a numpy array with shape specified by the reverse of sizes and dtype = ctype of the pointer
    '''
    shape = sizes[
        ::-1
    ]  # Reverse the sizes to convert from Fortran (column-major) to C/numpy (row-major) order
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    arr = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape)
    return arr
"""


class Backend(Enum):
    CPU = "run_gtfn"
    GPU = "run_gtfn_gpu"
    ROUNDTRIP = "run_roundtrip"


def parse_type_spec(type_spec: TypeSpec) -> tuple[list[Dimension], ScalarKind]:
    if isinstance(type_spec, ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, FieldType):
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def flatten_and_get_unique_elts(list_of_lists: list[list[str]]) -> list[str]:
    return sorted(set(item for sublist in list_of_lists for item in sublist))
