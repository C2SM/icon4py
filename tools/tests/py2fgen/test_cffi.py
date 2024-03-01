# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from cffi import FFI

from icon4pytools.py2fgen.plugin import unpack


@pytest.fixture
def ffi():
    return FFI()


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

    result = unpack(ptr, rows, cols)

    assert np.array_equal(result, expected_result)
