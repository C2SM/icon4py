# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-copy GT4Py Fields over ComIn buffers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from gt4py import next as gtx
from gt4py.next import common as gtx_common


def data_ptr(array: Any) -> int:
    """Return the address of the first element of a NumPy or CuPy array."""
    if isinstance(array, np.ndarray):
        return int(array.__array_interface__["data"][0])
    return int(array.data.ptr)


def field_view(array: Any, dims: Sequence[gtx.Dimension]) -> gtx.Field:
    """
    Return a zero-copy 'gtx.Field' over 'array', every domain starting at 0.

    This is the kernel of py2fgen's '_as_field' ('icon4py_export.py'). Unlike 'gtx.as_field',
    which copies, the Field shares the memory of 'array', so GT4Py programs write through to
    ICON's arrays. The check below makes a copying GT4Py fail loudly instead of silently
    losing the writes. '_field' is looked up at call time, so tests can replace it.
    """
    field = gtx_common._field(
        array, domain=gtx_common.domain(dict(zip(dims, array.shape, strict=True)))
    )
    if data_ptr(field.ndarray) != data_ptr(array) or field.ndarray.strides != array.strides:
        raise ValueError("'field_view': the Field does not alias the ComIn buffer.")
    return field
