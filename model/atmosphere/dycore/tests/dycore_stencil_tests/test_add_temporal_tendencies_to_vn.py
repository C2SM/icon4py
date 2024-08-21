# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestAddTemporalTendenciesToVn(StencilTest):
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
        vn_nnew = vn_nnow + dtime * (
            ddt_vn_apc_ntl1 + ddt_vn_phy - cpd * z_theta_v_e * z_gradh_exner
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid):
        dtime, cpd = wpfloat("10.0"), wpfloat("10.0")
        vn_nnow = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ddt_vn_apc_ntl1 = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        ddt_vn_phy = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_theta_v_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_gradh_exner = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn_nnew = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn_nnew=vn_nnew,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
