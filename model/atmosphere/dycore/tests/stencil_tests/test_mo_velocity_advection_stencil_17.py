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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_17 import (
    mo_velocity_advection_stencil_17,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, as_1D_sparse_field, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoVelocityAdvectionStencil17(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_17
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid, e_bln_c_s: np.array, z_v_grad_w: np.array, ddt_w_adv: np.array, **kwargs
    ) -> dict:
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        ddt_w_adv = ddt_w_adv + np.sum(
            z_v_grad_w[grid.connectivities[C2EDim]]
            * e_bln_c_s[grid.get_offset_provider("C2CE").table],
            axis=1,
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        z_v_grad_w = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        e_bln_c_s = as_1D_sparse_field(random_field(grid, CellDim, C2EDim, dtype=wpfloat), CEDim)
        ddt_w_adv = random_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_v_grad_w=z_v_grad_w,
            ddt_w_adv=ddt_w_adv,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
