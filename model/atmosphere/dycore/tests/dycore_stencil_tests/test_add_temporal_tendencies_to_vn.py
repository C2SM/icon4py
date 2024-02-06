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

from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def mo_solve_nonhydro_stencil_24_numpy(
    mesh,
    vn_nnow: np.array,
    ddt_vn_apc_ntl1: np.array,
    ddt_vn_phy: np.array,
    z_theta_v_e: np.array,
    z_gradh_exner: np.array,
    dtime: float,
    cpd: float,
) -> np.array:
    vn_nnew = vn_nnow + dtime * (ddt_vn_apc_ntl1 + ddt_vn_phy - cpd * z_theta_v_e * z_gradh_exner)
    return vn_nnew


class TestMoSolveNonhydroStencil24(StencilTest):
    PROGRAM = add_temporal_tendencies_to_vn
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        grid,
        vn_nnow: np.array,
        ddt_vn_apc_ntl1: np.array,
        ddt_vn_phy: np.array,
        z_theta_v_e: np.array,
        z_gradh_exner: np.array,
        dtime: float,
        cpd: float,
        **kwargs,
    ) -> dict:
        vn_nnew = mo_solve_nonhydro_stencil_24_numpy(
            grid,
            vn_nnow,
            ddt_vn_apc_ntl1,
            ddt_vn_phy,
            z_theta_v_e,
            z_gradh_exner,
            dtime,
            cpd,
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid):
        dtime, cpd = wpfloat("10.0"), wpfloat("10.0")
        vn_nnow = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        ddt_vn_apc_ntl1 = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        ddt_vn_phy = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_theta_v_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        z_gradh_exner = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn_nnew = zero_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn_nnew=vn_nnew,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
