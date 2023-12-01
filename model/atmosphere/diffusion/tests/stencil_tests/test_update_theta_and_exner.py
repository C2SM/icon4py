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

from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    update_theta_and_exner,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestUpdateThetaAndExner(StencilTest):
    PROGRAM = update_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        grid,
        z_temp: np.array,
        area: np.array,
        theta_v: np.array,
        exner: np.array,
        rd_o_cvd,
        **kwargs,
    ) -> tuple[np.array]:
        area = np.expand_dims(area, axis=0)
        z_theta = theta_v
        theta_v = theta_v + (np.expand_dims(area, axis=-1) * z_temp)
        exner = exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid):
        z_temp = random_field(grid, CellDim, KDim, dtype=vpfloat)
        area = random_field(grid, CellDim, dtype=wpfloat)
        theta_v = random_field(grid, CellDim, KDim, dtype=wpfloat)
        exner = random_field(grid, CellDim, KDim, dtype=wpfloat)
        rd_o_cvd = vpfloat("5.0")

        return dict(
            z_temp=z_temp,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
