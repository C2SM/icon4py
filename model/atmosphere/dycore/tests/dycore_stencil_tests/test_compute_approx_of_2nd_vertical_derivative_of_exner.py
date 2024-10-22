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

from icon4py.model.atmosphere.dycore.compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


class TestComputeApproxOf2ndVerticalDerivativeOfExner(StencilTest):
    PROGRAM = compute_approx_of_2nd_vertical_derivative_of_exner
    OUTPUTS = ("z_dexner_dz_c_2",)

    @staticmethod
    def reference(
        grid,
        z_theta_v_pr_ic: np.array,
        d2dexdz2_fac1_mc: np.array,
        d2dexdz2_fac2_mc: np.array,
        z_rth_pr_2: np.array,
        **kwargs,
    ) -> dict:
        z_theta_v_pr_ic_offset_1 = z_theta_v_pr_ic[:, 1:]
        z_dexner_dz_c_2 = -0.5 * (
            (z_theta_v_pr_ic[:, :-1] - z_theta_v_pr_ic_offset_1) * d2dexdz2_fac1_mc
            + z_rth_pr_2 * d2dexdz2_fac2_mc
        )
        return dict(z_dexner_dz_c_2=z_dexner_dz_c_2)

    @pytest.fixture
    def input_data(self, grid):
        z_theta_v_pr_ic = random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat
        )
        d2dexdz2_fac1_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        d2dexdz2_fac2_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        z_dexner_dz_c_2 = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            z_rth_pr_2=z_rth_pr_2,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
