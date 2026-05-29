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
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from icon4py.tools.py2fgen import _definitions


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi

C_STR_TYPE_TO_NP_DTYPE: Final[dict[str, np.dtype]] = {
    "int": np.dtype(np.int32),
    "double": np.dtype(np.float64),
    "float": np.dtype(np.float32),
    # see comment in `_codegen.BUILTIN_TO_CPP_TYPE` on why bool is `unsigned char` on the C side
    "unsigned char": np.dtype(np.bool_),
    "long": np.dtype(np.int64),
}


def _unpack_numpy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> np.typing.NDArray:
    """
    Converts a C pointer into a NumPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations requiring in-place modification of CPU data, enabling
    changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ffi:    A CFFI FFI instance.
        ptr:    A CFFI pointer to the beginning of the data array in CPU memory. This pointer
                should reference a contiguous block of memory whose total size matches the product
                of the specified dimensions.
        *sizes: Variable length argument list specifying the dimensions of the array.
                These sizes determine the shape of the resulting NumPy array.

    Returns:
        A NumPy array that provides a direct view of the data pointed to by `ptr`.
        This array shares the underlying data with the original Fortran code, allowing
        modifications made through the array to affect the original data.
    """
    length = math.prod(sizes)
    c_type = ffi.getctype(
        ffi.typeof(ptr).item
    )  # TODO(): use the type from the annotation and add a debug assert that they are fine

    # Map C data types to NumPy dtypes
    dtype = C_STR_TYPE_TO_NP_DTYPE.get(c_type)
    if dtype is None:
        raise ValueError(f"Unsupported C data type: {c_type}")

    # Create a NumPy array from the buffer, specifying the Fortran order
    arr = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(c_type)), dtype=dtype).reshape(  # type: ignore[call-overload]
        sizes, order="F"
    )
    return arr


def _unpack_cupy(ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int) -> cp.ndarray:
    """
    Converts a C pointer into a CuPy array to directly manipulate memory allocated in Fortran.
    This function is needed for operations that require in-place modification of GPU data,
    enabling changes made in Python to reflect immediately in the original Fortran memory space.

    Args:
        ffi:    A CFFI FFI instance.
        ptr:    A CFFI pointer to GPU memory allocated by OpenACC, representing
                the starting address of the data. This pointer must correspond to
                a contiguous block of memory whose total size matches the product
                of the specified dimensions.
        *sizes: Variable length argument list specifying the dimensions of the array.
                These sizes determine the shape of the resulting CuPy array.

    Returns:
        A CuPy array that provides a direct view of the data pointed to by `ptr`.
        This array shares the underlying data with the original Fortran code, allowing
        modifications made through the array to affect the original data.
    """
    assert cp is not None

    if not sizes:
        raise ValueError("Sizes must be provided to determine the array shape.")

    length = math.prod(sizes)
    c_type = ffi.getctype(ffi.typeof(ptr).item)

    dtype = C_STR_TYPE_TO_NP_DTYPE.get(c_type)
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


def unpack(
    xp: types.ModuleType, ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int
) -> np.typing.NDArray:
    if xp is np:
        return _unpack_numpy(ffi, ptr, *sizes)
    elif xp is cp:
        return _unpack_cupy(ffi, ptr, *sizes)
    else:
        raise ValueError(f"Unsupported array type: {xp}. Expected Numpy or CuPy.")


def as_array(ffi: cffi.FFI, array_info: _definitions.ArrayInfo) -> np.ndarray | None:  # or cupy
    """
    Utility function to convert an ArrayInfo to a NumPy or CuPy array.

    The returned array is a direct view of the Fortran-allocated memory; its
    element type is taken from the CFFI pointer.

    Args:
        ffi:        The CFFI FFI instance.
        array_info: The ArrayInfo object containing the pointer and shape information.
    """
    ptr, shape, on_gpu, is_optional = array_info
    xp = cp if on_gpu else np
    if ptr == ffi.NULL:
        if is_optional:
            return None
        else:
            raise RuntimeError("Parameter is not optional, but received 'NULL'.")
    return unpack(xp, ffi, ptr, *shape)


def _as_array_mapping() -> Callable[[_definitions.ArrayInfo, cffi.FFI], _definitions.NDArray]:
    # since we typically create one mapper per parameter, maxsize=2 is a good default for double buffering
    @functools.lru_cache(maxsize=2)
    def impl(array_info: _definitions.ArrayInfo, *, ffi: cffi.FFI) -> _definitions.NDArray:
        return as_array(ffi, array_info)

    return impl


def _int_to_bool(value: int, ffi: cffi.FFI) -> bool:
    return bool(value)


def default_mapping(
    _: Any, param_descriptor: _definitions.ParamDescriptor
) -> _definitions.MapperType | None:
    """
    Provide default mappings for raw Fortran data to Python data types.

    Array parameters are mapped from 'ArrayInfo's to NumPy/CuPy arrays.
    Scalar 'BOOL' is delivered by CFFI as a Python 'int' (the C side is
    `unsigned char`); convert it to 'bool' here.
    """
    if isinstance(param_descriptor, _definitions.ArrayParamDescriptor):
        # ArrayInfos to Numpy/CuPy arrays
        return _as_array_mapping()
    if (
        isinstance(param_descriptor, _definitions.ScalarParamDescriptor)
        and param_descriptor.dtype == _definitions.BOOL
    ):
        return _int_to_bool
    return None
