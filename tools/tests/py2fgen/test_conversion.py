# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.tools.py2fgen._conversion import _unpack_numpy


@pytest.fixture
def ffi():
    import cffi

    return cffi.FFI()


@pytest.mark.parametrize(
    "data, expected_result",
    [
        ([1.0, 2.0, 3.0, 4.0], np.array([[1.0, 3.0], [2.0, 4.0]])),
        ([1, 2, 3, 4], np.array([[1, 3], [2, 4]])),
    ],
)
def test_unpack_column_major(data, expected_result, ffi):
    ptr = ffi.new("double[]", data) if isinstance(data[0], float) else ffi.new("int[]", data)

    rows, cols = expected_result.shape

    result = _unpack_numpy(ffi, ptr, rows, cols)

    assert np.array_equal(result, expected_result)
