# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    update_theta_and_exner,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def update_theta_and_exner_numpy(
    mesh,
    z_temp: np.array,
    area: np.array,
    theta_v: np.array,
    exner: np.array,
    rd_o_cvd: float,
) -> tuple[np.array]:
    area = np.expand_dims(area, axis=0)
    z_theta = theta_v
    theta_v = theta_v + (np.expand_dims(area, axis=-1) * z_temp)
    exner = exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))
    return theta_v, exner


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
        rd_o_cvd: float,
        **kwargs,
    ) -> tuple[np.array]:
        theta_v, exner = update_theta_and_exner_numpy(grid, z_temp, area, theta_v, exner, rd_o_cvd)
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid):
        z_temp = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        area = random_field(grid, dims.CellDim, dtype=wpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rd_o_cvd = vpfloat("5.0")

        return dict(
            z_temp=z_temp,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
