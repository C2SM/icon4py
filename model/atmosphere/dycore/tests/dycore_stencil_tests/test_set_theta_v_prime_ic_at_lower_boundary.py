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

from icon4py.model.atmosphere.dycore.set_theta_v_prime_ic_at_lower_boundary import (
    set_theta_v_prime_ic_at_lower_boundary,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat

from .test_interpolate_to_surface import interpolate_to_surface_numpy


class TestInitThetaVPrimeIcAtLowerBoundary(StencilTest):
    PROGRAM = set_theta_v_prime_ic_at_lower_boundary
    OUTPUTS = ("z_theta_v_pr_ic", "theta_v_ic")

    @staticmethod
    def reference(
        grid,
        wgtfacq_c: np.array,
        z_rth_pr: np.array,
        theta_ref_ic: np.array,
        z_theta_v_pr_ic: np.array,
        theta_v_ic: np.array,
        **kwargs,
    ) -> dict:
        z_theta_v_pr_ic = interpolate_to_surface_numpy(
            grid=grid,
            wgtfacq_c=wgtfacq_c,
            interpolant=z_rth_pr,
            interpolation_to_surface=z_theta_v_pr_ic,
        )
        theta_v_ic[:, 3:] = (theta_ref_ic + z_theta_v_pr_ic)[:, 3:]
        return dict(z_theta_v_pr_ic=z_theta_v_pr_ic, theta_v_ic=theta_v_ic)

    @pytest.fixture
    def input_data(self, grid):
        wgtfacq_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_rth_pr = random_field(grid, CellDim, KDim, dtype=vpfloat)
        theta_ref_ic = random_field(grid, CellDim, KDim, dtype=vpfloat)
        z_theta_v_pr_ic = random_field(grid, CellDim, KDim, dtype=vpfloat)
        theta_v_ic = zero_field(grid, CellDim, KDim, dtype=wpfloat)

        return dict(
            wgtfacq_c=wgtfacq_c,
            z_rth_pr=z_rth_pr,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(3),
            vertical_end=int32(grid.num_levels),
        )
