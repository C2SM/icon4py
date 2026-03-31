# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def compute_perturbation_of_rho_and_theta_numpy(
    rho: np.ndarray,
    rho_ref_mc: np.ndarray,
    theta_v: np.ndarray,
    theta_ref_mc: np.ndarray,
) -> tuple[np.ndarray, ...]:
    z_rth_pr_1 = rho - rho_ref_mc
    z_rth_pr_2 = theta_v - theta_ref_mc
    return (z_rth_pr_1, z_rth_pr_2)


class TestComputePerturbationOfRhoAndTheta(StencilTest):
    PROGRAM = compute_perturbation_of_rho_and_theta
    OUTPUTS = ("z_rth_pr_1", "z_rth_pr_2")

    @static_reference
    def reference(
        grid: base.Grid,
        rho: np.ndarray,
        rho_ref_mc: np.ndarray,
        theta_v: np.ndarray,
        theta_ref_mc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (z_rth_pr_1, z_rth_pr_2) = compute_perturbation_of_rho_and_theta_numpy(
            rho=rho, rho_ref_mc=rho_ref_mc, theta_v=theta_v, theta_ref_mc=theta_ref_mc
        )
        return dict(z_rth_pr_1=z_rth_pr_1, z_rth_pr_2=z_rth_pr_2)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_ref_mc = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_ref_mc = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_1 = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_2 = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)

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
