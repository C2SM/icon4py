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

from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestComputePerturbationOfRhoAndTheta(StencilTest):
    PROGRAM = compute_perturbation_of_rho_and_theta
    OUTPUTS = ("z_rth_pr_1", "z_rth_pr_2")

    @staticmethod
    def reference(
        grid,
        rho: np.array,
        rho_ref_mc: np.array,
        theta_v: np.array,
        theta_ref_mc: np.array,
        **kwargs,
    ) -> tuple[np.array]:
        z_rth_pr_1 = rho - rho_ref_mc
        z_rth_pr_2 = theta_v - theta_ref_mc
        return dict(z_rth_pr_1=z_rth_pr_1, z_rth_pr_2=z_rth_pr_2)

    @pytest.fixture
    def input_data(self, grid):
        rho = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_ref_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_ref_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_1 = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_2 = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            rho=rho,
            rho_ref_mc=rho_ref_mc,
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
