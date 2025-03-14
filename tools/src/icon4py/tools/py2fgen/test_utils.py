# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional

import cffi
import numpy as np

from icon4py.tools.py2fgen import _codegen, _definitions


def to_np_dtype(dtype: _definitions.ScalarKind) -> np.dtype:
    if dtype == _definitions.ScalarKind.INT32:
        return np.dtype(np.int32)
    elif dtype == _definitions.ScalarKind.INT64:
        return np.dtype(np.int64)
    elif dtype == _definitions.ScalarKind.FLOAT32:
        return np.dtype(np.float32)
    elif dtype == _definitions.ScalarKind.FLOAT64:
        return np.dtype(np.float64)
    elif dtype == _definitions.ScalarKind.BOOL:
        return np.dtype(np.bool_)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def from_np_dtype(dtype: np.dtype) -> _definitions.ScalarKind:
    dtype = np.dtype(dtype)
    if dtype == np.int32:
        return _definitions.ScalarKind.INT32
    elif dtype == np.int64:
        return _definitions.ScalarKind.INT64
    elif dtype == np.float32:
        return _definitions.ScalarKind.FLOAT32
    elif dtype == np.float64:
        return _definitions.ScalarKind.FLOAT64
    elif dtype == np.bool_:
        return _definitions.ScalarKind.BOOL
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def array_descriptor(
    ptr: cffi.FFI.CData,
    shape: tuple[int, ...],
    on_gpu: bool,
    is_optional: bool,
) -> _definitions.ArrayDescriptor:
    return (ptr, shape, on_gpu, is_optional)


def array_to_array_descriptor(
    arr: np.ndarray,
    *,
    ffi: Optional[cffi.FFI] = None,
) -> _definitions.ArrayDescriptor:
    if ffi is None:
        ffi = cffi.FFI()
    # TODO(havogt): need to move bool handling to Fortran side
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int32, copy=True)

    addr = arr.ctypes.data
    strtype = _codegen.BUILTIN_TO_CPP_TYPE[from_np_dtype(arr.dtype)]
    ptr = ffi.cast(f"{strtype}*", addr)

    return array_descriptor(ptr, arr.shape, False, False)
