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

from icon4py.model.common import dimension as dims
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestMoIntpRbfRbfVecInterpolVertex(StencilTest):
    PROGRAM = mo_intp_rbf_rbf_vec_interpol_vertex
    OUTPUTS = ("p_u_out", "p_v_out")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        p_e_in: np.ndarray,
        ptr_coeff_1: np.ndarray,
        ptr_coeff_2: np.ndarray,
        **kwargs,
    ) -> tuple[np.array]:
        v2e = connectivities[dims.V2EDim]
        ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
        p_u_out = np.sum(p_e_in[v2e] * ptr_coeff_1, axis=1)

        ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
        p_v_out = np.sum(p_e_in[v2e] * ptr_coeff_2, axis=1)

        return dict(p_v_out=p_v_out, p_u_out=p_u_out)

    @pytest.fixture
    def input_data(self, grid):
        p_e_in = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ptr_coeff_1 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        ptr_coeff_2 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        p_v_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)
        p_u_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)

        return dict(
            p_e_in=p_e_in,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            p_v_out=p_v_out,
            p_u_out=p_u_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
