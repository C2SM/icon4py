# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray], p_cell_in: np.ndarray, c_intp: np.ndarray
) -> np.ndarray:
    v2c = connectivities[dims.V2C]
    c_intp = np.expand_dims(c_intp, axis=-1)
    p_vert_out = np.sum(np.where((v2c != -1)[:, :, np.newaxis], p_cell_in[v2c] * c_intp, 0), axis=1)
    return p_vert_out


class TestMoIconInterpolationScalarCells2vertsScalarRiDsl(stencil_tests.StencilTest):
    PROGRAM = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl
    OUTPUTS = ("p_vert_out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        p_cell_in: np.ndarray,
        c_intp: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        p_vert_out = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            connectivities, p_cell_in, c_intp
        )
        return dict(
            p_vert_out=p_vert_out,
        )

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        p_cell_in = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        c_intp = self.data_alloc.random_field(dims.VertexDim, dims.V2CDim, dtype=wpfloat)
        p_vert_out = self.data_alloc.zero_field(dims.VertexDim, dims.KDim, dtype=vpfloat)

        return dict(
            p_cell_in=p_cell_in,
            c_intp=c_intp,
            p_vert_out=p_vert_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
