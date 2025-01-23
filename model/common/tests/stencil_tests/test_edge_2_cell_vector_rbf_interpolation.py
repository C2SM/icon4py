# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers


class TestEdge2CellVectorRBFInterpolation(helpers.StencilTest):
    PROGRAM = edge_2_cell_vector_rbf_interpolation
    OUTPUTS = ("p_u_out", "p_v_out")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_e_in: np.array,
        ptr_coeff_1: np.array,
        ptr_coeff_2: np.array,
        **kwargs,
    ) -> dict:
        c2e2c2e = connectivities[dims.C2E2C2EDim]
        ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
        ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
        p_u_out = np.sum(p_e_in[c2e2c2e] * ptr_coeff_1, axis=1)
        p_v_out = np.sum(p_e_in[c2e2c2e] * ptr_coeff_2, axis=1)

        return dict(p_v_out=p_v_out, p_u_out=p_u_out)

    @pytest.fixture
    def input_data(self, grid):
        p_e_in = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        ptr_coeff_1 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2EDim, dtype=ta.wpfloat)
        ptr_coeff_2 = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2EDim, dtype=ta.wpfloat)
        p_v_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        p_u_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            p_e_in=p_e_in,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            p_v_out=p_v_out,
            p_u_out=p_u_out,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
