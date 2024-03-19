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

# necessary imports for generated code to work
import logging

import cupy as cp
import numpy as np

# all other imports from the module from which the function is being wrapped
from gt4py.next.ffront.fbuiltins import Field, float64, int32
from gt4py.next.iterator.embedded import np_as_located_field
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from identity_plugin import ffi
from numpy.typing import NDArray


log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"

logging.basicConfig(
    filename="py_cffi.log", level=logging.DEBUG, format=log_format, datefmt="%Y-%m-%d %H:%M:%S"
)

# We need a grid to pass offset providers
grid = SimpleGrid()


from icon4pytools.py2fgen.wrappers.simple import identity


def unpack(ptr, *sizes: int) -> NDArray:
    """
    Converts a C pointer pointing to data in Fortran (column-major) order into a NumPy array.

    This function facilitates the handling of numerical data shared between C (or Fortran) and Python,
    especially when the data originates from Fortran routines that use column-major storage order.
    It creates a NumPy array that directly views the data pointed to by `ptr`, without copying, and reshapes
    it according to the specified dimensions. The resulting NumPy array uses Fortran order ('F') to preserve
    the original data layout.

    Args:
        ptr (CData): A CFFI pointer to the beginning of the data array. This pointer should reference
            a contiguous block of memory whose total size matches the product of the specified dimensions.
        *sizes (int): Variable length argument list representing the dimensions of the array. The product
            of these sizes should match the total number of elements in the data block pointed to by `ptr`.

    Returns:
        np.ndarray: A NumPy array view of the data pointed to by `ptr`. The array will have the shape
        specified by `sizes` and the data type (`dtype`) corresponding to the C data type of `ptr`.
        The array is created with Fortran order to match the original column-major data layout.

    Note:
        The function does not perform any copying of the data. Modifications to the resulting NumPy array
        will affect the original data pointed to by `ptr`. Ensure that the lifetime of the data pointed to
        by `ptr` extends beyond the use of the returned NumPy array to prevent data corruption or access
        violations.
    """
    length = np.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    # Map C data types to NumPy dtypes
    dtype_map: dict[str, np.dtype] = {
        "int": np.dtype(np.int32),
        "double": np.dtype(np.float64),
    }
    dtype = dtype_map.get(c_type, np.dtype(c_type))

    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore
        sizes, order="F"
    )
    return arr


def unpack_gpu(ptr, *sizes: int) -> cp.ndarray:
    if not sizes:
        raise ValueError("Sizes must be provided to determine the array shape.")

    length = np.prod(sizes)
    # c_type = ffi.getctype(ffi.typeof(ptr).item)

    dtype_map = {
        "int": cp.int32,
        "double": cp.float64,
    }
    # dtype = dtype_map.get(c_type, None)
    dtype = cp.float64
    # if dtype is None:
    #     raise ValueError(f"Unsupported C data type: {c_type}")

    # itemsize = ffi.sizeof(c_type)
    itemsize = 8
    total_size = length * itemsize

    device_id = 0
    with cp.cuda.Device(device_id):
        print(ptr)
        ptr_val = int(ffi.cast("uintptr_t", ptr))
        print(ptr_val)
        mem = cp.cuda.UnownedMemory(ptr_val, total_size, owner=ptr, device_id=device_id)
        print(mem)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        print(memptr)

        # invalid memory access?
        arr = cp.ndarray(shape=sizes, dtype=dtype, memptr=memptr, order="F")
        print(arr)
        print(type(arr))

        # arr = np.frombuffer(ffi.buffer(ptr, total_size), dtype=dtype).reshape(  # type: ignore
        #     sizes, order="F"
        # )
        # print(type(arr))

        # # Convert the NumPy array to a CuPy array
        # cp_arr = cp.array(np_arr, copy=False)
        # print(type(cp_arr))

    return arr


def int_array_to_bool_array(int_array: NDArray) -> NDArray:
    """
    Converts a NumPy array of integers to a boolean array.
    In the input array, 0 represents False, and any non-zero value (1 or -1) represents True.

    Args:
        int_array: A NumPy array of integers.

    Returns:
        A NumPy array of booleans.
    """
    bool_array = int_array != 0
    return bool_array


def list_available_gpus():
    num_gpus = cp.cuda.runtime.getDeviceCount()
    logging.debug("Total GPUs available: %d" % num_gpus)

    for i in range(num_gpus):
        device = cp.cuda.Device(i)
        logging.debug(device)


@ffi.def_extern()
def identity_wrapper(inp: Field[[CellDim, KDim], float64], n_Cell: int32, n_K: int32):
    try:
        logging.info("Python Execution Context Start")

        list_available_gpus()

        # Unpack pointers into Ndarrays

        msg = "inp before unpacking: %s" % str(inp)
        logging.debug(msg)

        inp = unpack_gpu(inp, n_Cell, n_K)
        msg = "inp after unpacking: %s" % str(inp)
        logging.debug(msg)
        msg = "shape of inp after unpacking = %s" % str(inp.shape)
        logging.debug(msg)

        # Allocate GT4Py Fields

        inp = np_as_located_field(CellDim, KDim)(inp)
        msg = "shape of inp after allocating as field = %s" % str(inp.shape)
        logging.debug(msg)
        msg = "inp after allocating as field: %s" % str(inp.ndarray)
        logging.debug(msg)

        identity(inp)

        # debug info

        msg = "shape of inp after computation = %s" % str(inp.shape)
        logging.debug(msg)
        msg = "inp after computation: %s" % str(inp.ndarray)
        logging.debug(msg)

        logging.info("Python Execution Context End")

    except Exception as e:
        logging.exception(f"A Python error occurred: {e}")
        return 1

    return 0
