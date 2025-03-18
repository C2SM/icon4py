# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Optional

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
) -> _definitions.ArrayInfo:
    return (ptr, shape, on_gpu, is_optional)


def array_to_array_descriptor(
    arr: np.ndarray,
    *,
    ffi: Optional[cffi.FFI] = None,
    keep_alive: bool = True,
    as_fortran_layout: bool = True,
) -> _definitions.ArrayInfo:
    if ffi is None:
        ffi = cffi.FFI()
    # TODO(havogt): need to move bool handling to Fortran side
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int32, copy=True)
    if as_fortran_layout and not arr.flags["F_CONTIGUOUS"]:
        arr = np.asfortranarray(arr)

    addr = arr.ctypes.data
    strtype = _codegen.BUILTIN_TO_CPP_TYPE[from_np_dtype(arr.dtype)]
    ptr = ffi.cast(f"{strtype}*", addr)

    # bind the lifetime of `arr`to `ptr`
    def nothing(_: Any) -> None:
        pass

    if keep_alive:
        ptr = ffi.gc(ptr, lambda _: nothing(arr))

    return array_descriptor(ptr, arr.shape, False, False)
