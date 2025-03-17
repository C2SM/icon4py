# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import pytest

from icon4py.tools.py2fgen._conversion import _int_array_to_bool_array, unpack


try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.fixture
def ffi():
    import cffi

    return cffi.FFI()


@pytest.fixture(
    params=[
        np,
        pytest.param(
            cp,
            # TODO(havogt): move to `requires_gpu` pattern?
            marks=pytest.mark.skipif(
                os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI."
            ),
        ),
    ],
    ids=["numpy", "cupy"],
)
def xp(request):
    return request.param


def _get_ptr(xp, arr):
    if xp == np:
        return arr.ctypes.data
    else:
        return arr.data.ptr


@pytest.mark.parametrize(
    "ctype, rawdata, expected",
    [
        ("double", [1.0, 2.0, 3.0, 4.0], [[1.0, 3.0], [2.0, 4.0]]),
        ("long", [1, 2, 3, 4], [[1, 3], [2, 4]]),
    ],
)
def test_unpack_column_major(xp, ctype, rawdata, expected, ffi):
    expected_result = xp.array(expected)

    arr = xp.array(rawdata)
    ptr = ffi.cast(f"{ctype}*", _get_ptr(xp, arr))

    rows, cols = expected_result.shape

    result = unpack(xp, ffi, ptr, rows, cols)

    assert np.array_equal(result, expected_result)


def test_int_array_to_bool():
    testee = np.array([[0, -1, 1], [0, -1, 1]], dtype=np.int32, order="F")
    expected = np.array([[False, True, True], [False, True, True]], dtype=np.bool_, order="F")

    result = _int_array_to_bool_array(testee)
    assert result.flags["F_CONTIGUOUS"]
    assert np.array_equal(result, expected)
