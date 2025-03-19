# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gc

import cffi
import numpy as np
import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen.test_utils import array_to_array_info, from_np_dtype


@pytest.fixture
def ffi():
    return cffi.FFI()


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64])
def test_array_as_array_info(ffi, dtype):
    testee = np.array([0, 1, 2, 3, 4], dtype=dtype)

    result = array_to_array_info(testee, ffi=ffi)

    assert isinstance(result, tuple)
    arr = py2fgen.as_array(ffi, result, from_np_dtype(dtype))
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, testee)


def test_array_as_array_info_lifetime(ffi):
    result = array_to_array_info(np.arange(100000, dtype=np.int32), ffi=ffi)

    gc.collect()  # Force garbage collection
    _dummy = np.arange(100000, dtype=np.int32) + 1  # let's hope we get the same memory

    arr = py2fgen.as_array(ffi, result, from_np_dtype(np.int32))
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.arange(100000, dtype=np.int32))
