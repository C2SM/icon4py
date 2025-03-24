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
from typing import TYPE_CHECKING, Any, Callable, Final, Optional

import numpy as np

from icon4py.tools.py2fgen import _definitions


try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi

C_STR_TYPE_TO_NP_DTYPE: Final[dict[str, np.dtype]] = {
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
    # bool_array.flags.writeable = False # TODO(havogt): np.ndarray.__dlpack__() doesn't like the readonly flag # noqa: ERA001
    return bool_array


def unpack(
    xp: types.ModuleType, ffi: cffi.FFI, ptr: cffi.FFI.CData, *sizes: int
) -> np.typing.NDArray:
    if xp is np:
        return _unpack_numpy(ffi, ptr, *sizes)
    elif xp is cp:
        return _unpack_cupy(ffi, ptr, *sizes)
    else:
        raise ValueError(f"Unsupported array type: {xp}. Expected Numpy or CuPy.")


def as_array(
    ffi: cffi.FFI, array_info: _definitions.ArrayInfo, dtype: _definitions.ScalarKind
) -> Optional[np.ndarray]:  # or cupy
    """
    Utility function to convert an ArrayInfo to a NumPy or CuPy array.

    Boolean arrays are converted to a NumPy array of dtype 'bool'.

    Args:
        ffi:        The CFFI FFI instance.
        array_info: The ArrayInfo object containing the pointer and shape information.
        dtype:      The data type of the array.
                    Note, the Fortran/C type is already included 'ArrayInfo', however
                    for booleans, the ArrayInfo.dtype is 'int32', this 'dtype' should be 'BOOL'.
    """
    ptr, shape, on_gpu, is_optional = array_info
    xp = cp if on_gpu else np
    if ptr == ffi.NULL:
        if is_optional:
            return None
        else:
            raise RuntimeError("Parameter is not optional, but received 'NULL'.")
    arr = unpack(xp, ffi, ptr, *shape)
    if dtype == _definitions.BOOL:
        # TODO(havogt): This transformation breaks if we want to write to this array as we do a copy.
        # Probably we need to do this transformation by hand on the Fortran side and pass responsibility to the user.
        arr = _int_array_to_bool_array(arr)
    return arr


def _as_array_mapping(
    dtype: _definitions.ScalarKind,
) -> Callable[[_definitions.ArrayInfo, cffi.FFI], _definitions.NDArray]:
    # since we typically create one mapper per parameter, maxsize=2 is a good default for double buffering
    @functools.lru_cache(maxsize=2)
    def impl(array_info: _definitions.ArrayInfo, *, ffi: cffi.FFI) -> _definitions.NDArray:
        return as_array(ffi, array_info, dtype)

    return impl


def _int_to_bool(x: int, ffi: cffi.FFI) -> bool:
    return x != 0


def default_mapping(
    _: Any, param_descriptor: _definitions.ParamDescriptor
) -> _definitions.MapperType | None:
    """
    Provide default mappings for raw Fortran data to Python data types.
    The default mapping provides mapping functions for 'ArrayInfo's to NumPy/CuPy arrays
    and scalar bools (represented as 'int32') to Python bools.
    """
    if isinstance(param_descriptor, _definitions.ArrayParamDescriptor):
        # ArrayInfos to Numpy/CuPy arrays
        return _as_array_mapping(param_descriptor.dtype)
    if (
        isinstance(param_descriptor, _definitions.ScalarParamDescriptor)
        and param_descriptor.dtype == _definitions.BOOL
    ):
        # bools are passed as int32, convert to bool
        return _int_to_bool
    return None
