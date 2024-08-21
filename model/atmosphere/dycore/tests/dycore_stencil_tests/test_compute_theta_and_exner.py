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

from icon4py.model.atmosphere.dycore.compute_theta_and_exner import compute_theta_and_exner
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask
from icon4py.model.common.type_alias import wpfloat


class TestComputeThetaAndExner(StencilTest):
    PROGRAM = compute_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        grid,
        bdy_halo_c: np.array,
        rho: np.array,
        theta_v: np.array,
        exner: np.array,
        rd_o_cvd: float,
        rd_o_p0ref: float,
        **kwargs,
    ) -> dict:
        bdy_halo_c = np.expand_dims(bdy_halo_c, axis=-1)

        theta_v = np.where(bdy_halo_c == 1, exner, theta_v)
        exner = np.where(
            bdy_halo_c == 1, np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * exner)), exner
        )

        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid):
        rd_o_cvd = wpfloat("10.0")
        rd_o_p0ref = wpfloat("20.0")
        bdy_halo_c = random_mask(grid, dims.CellDim)
        exner = random_field(grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=wpfloat)
        rho = random_field(grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=wpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=wpfloat)

        return dict(
            bdy_halo_c=bdy_halo_c,
            rho=rho,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            rd_o_p0ref=rd_o_p0ref,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
