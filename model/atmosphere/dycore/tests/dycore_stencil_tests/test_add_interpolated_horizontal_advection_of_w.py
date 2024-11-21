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

from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, as_1D_sparse_field, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def add_interpolated_horizontal_advection_of_w_numpy(
    grid, e_bln_c_s: np.array, z_v_grad_w: np.array, ddt_w_adv: np.array, **kwargs
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    c2ce = grid.get_offset_provider("C2CE").table

    ddt_w_adv = ddt_w_adv + np.sum(
        z_v_grad_w[grid.connectivities[dims.C2EDim]] * e_bln_c_s[c2ce],
        axis=1,
    )
    return ddt_w_adv


class TestAddInterpolatedHorizontalAdvectionOfW(StencilTest):
    PROGRAM = add_interpolated_horizontal_advection_of_w
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid, e_bln_c_s: np.array, z_v_grad_w: np.array, ddt_w_adv: np.array, **kwargs
    ) -> dict:
        ddt_w_adv = add_interpolated_horizontal_advection_of_w_numpy(
            grid, e_bln_c_s, z_v_grad_w, ddt_w_adv
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        z_v_grad_w = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        e_bln_c_s = as_1D_sparse_field(
            random_field(grid, dims.CellDim, dims.C2EDim, dtype=wpfloat), dims.CEDim
        )
        ddt_w_adv = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_v_grad_w=z_v_grad_w,
            ddt_w_adv=ddt_w_adv,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
