# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import math
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from icon4py.tools.py2fgen import _definitions, utils


try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi

C_STR_TYPE_TO_NP_DTYPE: dict[str, np.dtype] = {
    "int": np.dtype(np.int32),
    "double": np.dtype(np.float64),
    "float": np.dtype(np.float32),
    "bool": np.dtype(np.bool_),
    "long": np.dtype(np.int64),
}


def _unpack_numpy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> np.typing.NDArray:
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

    dtype = C_STR_TYPE_TO_NP_DTYPE.get(c_type, np.dtype(c_type))

    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore
        sizes, order="F"
    )
    return arr


def _unpack_cupy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> cp.ndarray:
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

    dtype = C_STR_TYPE_TO_NP_DTYPE.get(c_type, None)
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
    xp = np if isinstance(int_array, np.ndarray) else cp
    bool_array = xp.array(int_array != 0, order="F", dtype=np.bool_)
    # bool_array.flags.writeable = False # TODO np.ndarray.__dlpack__() doesn't like the readonly flag # noqa: ERA001
    return bool_array


def unpack(xp: ModuleType, ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> np.typing.NDArray:
    if xp == np:
        return _unpack_numpy(ffi, ptr, *sizes)
    elif xp == cp:
        return _unpack_cupy(ffi, ptr, *sizes)
    else:
        raise ValueError(f"Unsupported array type: {xp}. Expected Numpy or CuPy.")


def _as_array(
    dtype: _definitions.ScalarKind,
) -> Callable[[_definitions.ArrayInfo, cffi.FFI], _definitions.NDArray]:
    # since we typically create one mapper per parameter, maxsize=2 is a good default for double buffering
    @functools.lru_cache(maxsize=2)
    def impl(array_descriptor: _definitions.ArrayInfo, *, ffi: cffi.FFI) -> _definitions.NDArray:
        if array_descriptor[3] and array_descriptor is None:
            return None
        return utils.as_array(ffi, array_descriptor, dtype)

    return impl


def _int_to_bool(x: int, ffi: cffi.FFI) -> bool:
    return x != 0


def default_mapping(
    _: Any, param_descriptor: _definitions.ParamDescriptor
) -> _definitions.MapperType | None:
    if isinstance(param_descriptor, _definitions.ArrayParamDescriptor):
        # ArrayInfos to Numpy/CuPy arrays
        return _as_array(param_descriptor.dtype)
    if (
        isinstance(param_descriptor, _definitions.ScalarParamDescriptor)
        and param_descriptor.dtype == _definitions.BOOL
    ):
        # bools are passed as int32, convert to bool
        return _int_to_bool
    return None
