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

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _conversion, test_utils


try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.fixture
def ffi():
    import cffi

    return cffi.FFI()


@pytest.fixture
def no_ffi_dummy():
    """
    Use to indicate that FFI is not used in the test.

    ... but where an FFI argument is required in the general case."""
    return None


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

    result = _conversion.unpack(xp, ffi, ptr, rows, cols)
    assert isinstance(result, xp.ndarray)
    assert xp.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "ctype, rawdtype, rawdata, expected",
    [
        ("double", np.float64, [1.0, 2.0, 3.0, 4.0], [[1.0, 3.0], [2.0, 4.0]]),
        ("long", np.int64, [1, 2, 3, 4], [[1, 3], [2, 4]]),
        ("int", np.int32, [-1, 0, -1, 0, 0, -1], [[True, True, False], [False, False, True]]),
    ],
)
def test_as_array(xp, ctype, rawdtype, rawdata, expected, ffi):
    expected_result = xp.array(expected)

    arr = xp.array(rawdata, dtype=rawdtype)
    ptr = ffi.cast(f"{ctype}*", _get_ptr(xp, arr))
    array_info = test_utils.array_info(
        ptr, shape=expected_result.shape, on_gpu=xp != np, is_optional=False
    )

    result = _conversion.as_array(ffi, array_info, test_utils.from_np_dtype(expected_result.dtype))

    assert isinstance(result, xp.ndarray)
    assert xp.array_equal(result, expected_result)


def test_int_array_to_bool():
    testee = np.array([[0, -1, 1], [0, -1, 1]], dtype=np.int32, order="F")
    expected = np.array([[False, True, True], [False, True, True]], dtype=np.bool_, order="F")

    result = _conversion._int_array_to_bool_array(testee)
    assert result.flags["F_CONTIGUOUS"]
    assert np.array_equal(result, expected)


def test_default_mapping_hook_array(ffi):
    array_ptr = ffi.new("int[10]")

    array_mapper = _conversion.default_mapping(
        None,
        py2fgen.ArrayParamDescriptor(
            rank=1, dtype=py2fgen.INT32, memory_space=py2fgen.MemorySpace.HOST, is_optional=False
        ),
    )
    result = array_mapper(
        test_utils.array_info(ptr=array_ptr, shape=(10,), on_gpu=False, is_optional=False),
        ffi=ffi,
    )

    assert isinstance(result, np.ndarray)


def test_default_mapping_hook_bool(no_ffi_dummy):
    bool_mapper = _conversion.default_mapping(
        None, py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL)
    )

    true_result = bool_mapper(-1, ffi=no_ffi_dummy)
    assert isinstance(true_result, bool)
    assert true_result
    false_result = bool_mapper(0, ffi=no_ffi_dummy)
    assert isinstance(false_result, bool)
    assert not false_result
