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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_16 import (
    mo_velocity_advection_stencil_16,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoVelocityAdvectionStencil16(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_16
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid,
        z_w_con_c: np.array,
        w: np.array,
        coeff1_dwdz: np.array,
        coeff2_dwdz: np.array,
        **kwargs,
    ) -> dict:
        ddt_w_adv = np.zeros_like(coeff1_dwdz)
        ddt_w_adv[:, 1:] = -z_w_con_c[:, 1:] * (
            w[:, :-2] * coeff1_dwdz[:, 1:]
            - w[:, 2:] * coeff2_dwdz[:, 1:]
            + w[:, 1:-1] * (coeff2_dwdz[:, 1:] - coeff1_dwdz[:, 1:])
        )
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        w = random_field(grid, CellDim, KDim, extend={KDim: 1}, dtype=wpfloat)
        coeff1_dwdz = random_field(grid, CellDim, KDim, dtype=vpfloat)
        coeff2_dwdz = random_field(grid, CellDim, KDim, dtype=vpfloat)
        ddt_w_adv = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            z_w_con_c=z_w_con_c,
            w=w,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            ddt_w_adv=ddt_w_adv,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
