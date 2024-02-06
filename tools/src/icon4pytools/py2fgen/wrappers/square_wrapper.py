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
# mypy: ignore-errors
# TODO(samkellerhals): Delete file once we can generate wrapper functions for programs. Use this for tests potentially.

# flake8: noqa D104
import numpy as np
from gt4py.next.ffront.fbuiltins import Field, float64, int32
from icon4py.model.common.dimension import CellDim, KDim

from icon4pytools.py2fgen.wrappers.square_functions import square_output_param


# TODO [Magdalena]: generalize this and provide it as decorator to def_extern functions?
# TODO [Magdalena]: generalize for arbitrary number of input fields
def unpack(ptr, *sizes) -> np.ndarray:
    """
    Unpacks an n-dimensional Fortran (column-major) array into a numpy array (row-major).

    :param ptr: c_pointer to the field
    :param sizes: variable number of arguments representing the dimensions of the array in Fortran order
    :return: a numpy array with shape specified by the reverse of sizes and dtype = ctype of the pointer
    """
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


def pack(ptr, arr: np.ndarray):
    """
    memcopies a numpy array into a pointer.

    :param ptr: c pointer
    :param arr: numpy array
    :return:
    """
    # for now only 2d
    length = np.prod(arr.shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ffi.memmove(ptr, np.ravel(arr), length * ffi.sizeof(c_type))


def square_wrapper(
    field_ptr: Field[[CellDim, KDim], float64],
    result_ptr: Field[[CellDim, KDim], float64],
    n_cell: int32,
    n_k: int32,
):
    """
    simple python function that squares all entries of a field of
    size nx x ny and returns a pointer to the result.

    :param field_ptr:
    :param nx:
    :param ny:
    :param result_ptr:
    :return:
    """
    a = unpack(field_ptr, n_cell, n_k)
    res = unpack(result_ptr, n_cell, n_k)
    print(res)
    print(a)
    square_output_param(a, res)
