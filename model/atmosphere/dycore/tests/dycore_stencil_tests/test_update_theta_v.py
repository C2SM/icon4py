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

from icon4py.model.atmosphere.dycore.update_theta_v import update_theta_v
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask
from icon4py.model.common.type_alias import wpfloat


class TestUpdateThetaV(StencilTest):
    PROGRAM = update_theta_v
    OUTPUTS = ("theta_v_new",)

    @staticmethod
    def reference(
        grid,
        mask_prog_halo_c: np.array,
        rho_now: np.array,
        theta_v_now: np.array,
        exner_new: np.array,
        exner_now: np.array,
        rho_new: np.array,
        theta_v_new: np.array,
        cvd_o_rd: float,
        **kwargs,
    ) -> dict:
        mask_prog_halo_c = np.expand_dims(mask_prog_halo_c, axis=-1)

        theta_v_new = np.where(
            mask_prog_halo_c,
            rho_now * theta_v_now * ((exner_new / exner_now - 1) * cvd_o_rd + 1.0) / rho_new,
            theta_v_new,
        )
        return dict(theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, grid):
        mask_prog_halo_c = random_mask(grid, CellDim)
        rho_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_v_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_new = random_field(grid, CellDim, KDim, dtype=wpfloat)
        exner_now = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rho_new = random_field(grid, CellDim, KDim, dtype=wpfloat)
        theta_v_new = random_field(grid, CellDim, KDim, dtype=wpfloat)
        cvd_o_rd = wpfloat("10.0")

        return dict(
            mask_prog_halo_c=mask_prog_halo_c,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_new=exner_new,
            exner_now=exner_now,
            rho_new=rho_new,
            theta_v_new=theta_v_new,
            cvd_o_rd=cvd_o_rd,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
