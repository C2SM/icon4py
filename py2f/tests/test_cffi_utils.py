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

import cffi
import numpy as np
import pytest
from functional.common import Field
from functional.ffront.fbuiltins import float32, float64, int32, int64

from icon4py.common.dimension import KDim, VertexDim
from icon4py.py2f.cffi_utils import UnknownDimensionException, to_fields
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field


mesh = SimpleMesh()


def test_unpack_args_from_buffer_to_field():
    @to_fields(dim_sizes={VertexDim: mesh.n_vertices, KDim: mesh.k_level})
    def identity(
        f1: Field[[VertexDim, KDim], float], factor: float
    ) -> tuple[float, Field[[VertexDim, KDim], float]]:
        return factor, f1

    ffi = cffi.FFI()
    input_array = np.asarray(random_field(mesh, VertexDim, KDim))
    input_factor = 0.5
    res_factor, result_field = identity(
        ffi.from_buffer("float*", input_array), input_factor
    )
    assert res_factor == input_factor
    assert np.allclose(np.asarray(result_field), input_array)


def test_only_scalar_args():
    @to_fields(dim_sizes={})
    def multiply(f1: float, f2: float32, f3: float64, i3: int64, i2: int32, i1: int):
        return f1 * f2 * f3, i1 * i2 * i3

    f_res, i_res = multiply(0.5, 2.0, 3.0, 1, 2, 3)
    assert f_res == 3.0
    assert i_res == 6


def test_unknwon_dimension_raises_exception():
    @to_fields(dim_sizes={})
    def do_nothing(f1: Field[[VertexDim], float]):
        pass

    input_array = np.asarray(random_field(mesh, VertexDim, KDim))
    with pytest.raises(UnknownDimensionException, match=r"size of dimension "):
        do_nothing(input_array)
