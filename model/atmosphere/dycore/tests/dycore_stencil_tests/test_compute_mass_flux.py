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

from icon4py.model.atmosphere.dycore.stencils.compute_mass_flux import compute_mass_flux
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputeMassFlux(StencilTest):
    PROGRAM = compute_mass_flux
    OUTPUTS = ("mass_fl_e", "z_theta_v_fl_e")

    @staticmethod
    def reference(
        grid,
        z_rho_e: np.array,
        z_vn_avg: np.array,
        ddqz_z_full_e: np.array,
        z_theta_v_e: np.array,
        **kwargs,
    ) -> dict:
        mass_fl_e = z_rho_e * z_vn_avg * ddqz_z_full_e
        z_theta_v_fl_e = mass_fl_e * z_theta_v_e
        return dict(mass_fl_e=mass_fl_e, z_theta_v_fl_e=z_theta_v_fl_e)

    @pytest.fixture
    def input_data(self, grid):
        z_rho_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_vn_avg = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ddqz_z_full_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        mass_fl_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_theta_v_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_theta_v_fl_e = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            z_rho_e=z_rho_e,
            z_vn_avg=z_vn_avg,
            ddqz_z_full_e=ddqz_z_full_e,
            z_theta_v_e=z_theta_v_e,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
