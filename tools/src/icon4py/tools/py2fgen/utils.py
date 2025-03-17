# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from icon4py.tools.py2fgen import _conversion, _definitions


try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

if TYPE_CHECKING:
    import cffi


def as_array(
    ffi: cffi.FFI, array_descriptor: _definitions.ArrayDescriptor, dtype: _definitions.ScalarKind
) -> Optional[np.ndarray]:  # or cupy
    unpack = _conversion._unpack_cupy if array_descriptor[2] else _conversion._unpack_numpy
    if array_descriptor[0] == ffi.NULL:
        if array_descriptor[3]:
            return None
        else:
            raise RuntimeError("Parameter is not optional, but received 'NULL'.")
    arr = unpack(ffi, array_descriptor[0], *array_descriptor[1])
    if dtype == _definitions.BOOL:
        # TODO(havogt): This transformation breaks if we want to write to this array as we do a copy.
        # Probably we need to do this transformation by hand on the Fortran side and pass responsibility to the user.
        arr = _conversion._int_array_to_bool_array(arr)
    return arr
