# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import TYPE_CHECKING, Optional, TypeAlias

import numpy as np
from gt4py.next.type_system import type_specifications as ts


try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi

    # Note, we use this plain tuple for performance.
    ArrayDescriptor: TypeAlias = tuple[cffi.FFI.CData, tuple[int, ...], bool, bool]
else:
    from typing import Any

    ArrayDescriptor: TypeAlias = tuple[Any, tuple[int, ...], bool, bool]


def _unpack(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> np.typing.NDArray:
    """
    Converts a C pointer into a NumPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations requiring in-place modification of CPU data, enabling
    changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ffi (cffi.FFI): A CFFI FFI instance.
        ptr (CData): A CFFI pointer to the beginning of the data array in CPU memory. This pointer
                     should reference a contiguous block of memory whose total size matches the product
                     of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting NumPy array.

    Returns:
        np.ndarray: A NumPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
    """
    length = math.prod(sizes)
    c_type = ffi.getctype(
        ffi.typeof(ptr).item
    )  # TODO use the type from the annotation and add a debug assert that they are fine

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


def _unpack_gpu(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> cp.ndarray:
    """
    Converts a C pointer into a CuPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations that require in-place modification of GPU data,
    enabling changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ffi (cffi.FFI): A CFFI FFI instance.
        ptr (cffi.CData): A CFFI pointer to GPU memory allocated by OpenACC, representing
                          the starting address of the data. This pointer must correspond to
                          a contiguous block of memory whose total size matches the product
                          of the specified dimensions.
        *sizes (int): Variable length argument list specifying the dimensions of the array.
                      These sizes determine the shape of the resulting CuPy array.

    Returns:
        cp.ndarray: A CuPy array that provides a direct view of the data pointed to by `ptr`.
                    This array shares the underlying data with the original Fortran code, allowing
                    modifications made through the array to affect the original data.
    """
    assert cp is not None

    if not sizes:
        raise ValueError("Sizes must be provided to determine the array shape.")

    length = math.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    dtype_map = {
        "int": cp.int32,
        "double": cp.float64,
    }
    dtype = dtype_map.get(c_type, None)
    if dtype is None:
        raise ValueError(f"Unsupported C data type: {c_type}")

    itemsize = ffi.sizeof(c_type)
    total_size = length * itemsize

    # cupy array from OpenACC device pointer
    current_device = cp.cuda.Device()
    ptr_val = int(ffi.cast("uintptr_t", ptr))
    mem = cp.cuda.UnownedMemory(ptr_val, total_size, owner=ptr, device_id=current_device.id)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr = cp.ndarray(shape=sizes, dtype=dtype, memptr=memptr, order="F")
    return arr


def _int_array_to_bool_array(int_array: np.typing.NDArray) -> np.typing.NDArray:
    """
    Converts a NumPy array of integers to a boolean array.
    In the input array, 0 represents False, and any non-zero value (1 or -1) represents True.

    Args:
        int_array: A NumPy array of integers.

    Returns:
        A NumPy array of booleans.
    """
    bool_array = int_array != 0
    bool_array.flags.writeable = False
    return bool_array


def as_array(
    ffi: cffi.FFI, array_descriptor: ArrayDescriptor, scalar_kind: ts.ScalarKind
) -> Optional[np.ndarray]:  # or cupy
    unpack = _unpack_gpu if array_descriptor[2] else _unpack
    if array_descriptor[0] == ffi.NULL:
        if array_descriptor[3]:
            return None
        else:
            raise RuntimeError("Parameter is not optional, but received 'NULL'.")
    arr = unpack(ffi, array_descriptor[0], *array_descriptor[1])
    if scalar_kind == ts.ScalarKind.BOOL:
        # TODO(havogt): This transformation breaks if we want to write to this array as we do a copy.
        # Probably we need to do this transformation by hand on the Fortran side and pass responsibility to the user.
        arr = _int_array_to_bool_array(arr)
    return arr
