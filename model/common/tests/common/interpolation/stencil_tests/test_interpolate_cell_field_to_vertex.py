# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.type_alias as types
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_vertex import (
    _interpolate_cell_field_to_vertex,
)
from icon4py.model.testing import reference_funcs, stencil_tests


@pytest.mark.skip_value_error
class TestInterpolateCellFieldToVertex(stencil_tests.StencilTest):
    PROGRAM = _interpolate_cell_field_to_vertex
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        cell_in: np.ndarray,
        c_int: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        return dict(
            out=reference_funcs.interpolate_cell_field_to_vertex_numpy(
                connectivities, cell_in, c_int
            ),
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        cell_in = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=types.wpfloat)
        c_int = data_alloc.random_field(dims.VertexDim, dims.V2CDim, dtype=types.wpfloat)
        return dict(
            cell_in=cell_in,
            c_int=c_int,
            out=data_alloc.zero_field(dims.VertexDim, dims.KHalfDim, dtype=types.wpfloat),
            domain={
                dims.VertexDim: (0, gtx.int32(grid.num_vertices)),
                dims.KHalfDim: (0, gtx.int32(grid.num_levels + 1)),
            },
        )
