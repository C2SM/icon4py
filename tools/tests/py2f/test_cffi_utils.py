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
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import float32, float64, int32, int64
from icon4py.model.common.dimension import E2CDim, EdgeDim, KDim, VertexDim

from icon4pytools.py2f.cffi_utils import UnknownDimensionException, to_fields

n_vertices = 9
n_edges = 27
levels = 12
e2c_sparse_size = 2


def random(*sizes):
    return np.random.default_rng().uniform(size=sizes)


@pytest.mark.parametrize("pointer_type", ["float*", "double*"])
def test_unpack_from_buffer_to_field(pointer_type: str):
    @to_fields(dim_sizes={VertexDim: n_vertices, KDim: levels})
    def identity(
        f1: Field[[VertexDim, KDim], float], factor: float
    ) -> tuple[float, Field[[VertexDim, KDim], float]]:
        return factor, f1

    ffi = cffi.FFI()
    input_array = random(n_vertices, levels)
    input_factor = 0.5
    res_factor, result_field = identity(ffi.from_buffer(pointer_type, input_array), input_factor)
    assert res_factor == input_factor
    assert np.allclose(result_field.asnumpy(), input_array)


def test_unpack_only_scalar_args():
    @to_fields(dim_sizes={})
    def multiply(f1: float, f2: float32, f3: float64, i3: int64, i2: int32, i1: int):
        return f1 * f2 * f3, i1 * i2 * i3

    f_res, i_res = multiply(0.5, 2.0, 3.0, 1, 2, 3)
    assert f_res == 3.0
    assert i_res == 6


@pytest.mark.parametrize("field_type", [int32, int, int64])
def test_unpack_local_field(field_type):
    ffi = cffi.FFI()

    @to_fields(dim_sizes={EdgeDim: n_edges, E2CDim: e2c_sparse_size})
    def local_field(f1: Field[[EdgeDim, E2CDim], field_type]):
        return f1

    input_field = np.arange(n_edges * e2c_sparse_size).reshape((n_edges, e2c_sparse_size))
    res_field = local_field(ffi.from_buffer("int*", input_field))
    assert np.allclose(res_field.asnumpy(), input_field, atol=0)


def test_unknown_dimension_raises_exception():
    @to_fields(dim_sizes={})
    def do_nothing(f1: Field[[VertexDim], float]):
        pass

    input_array = random()
    with pytest.raises(UnknownDimensionException, match=r"size of dimension "):
        do_nothing(input_array)
