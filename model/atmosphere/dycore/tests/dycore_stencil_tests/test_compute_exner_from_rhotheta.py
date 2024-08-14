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

from icon4py.model.atmosphere.dycore.compute_exner_from_rhotheta import compute_exner_from_rhotheta
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import wpfloat


class TestComputeExnerFromRhotheta(StencilTest):
    PROGRAM = compute_exner_from_rhotheta
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        grid,
        rho: np.array,
        exner: np.array,
        rd_o_cvd: float,
        rd_o_p0ref: float,
        **kwargs,
    ) -> dict:
        theta_v = np.copy(exner)
        exner = np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * theta_v))
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid):
        rd_o_cvd = wpfloat("10.0")
        rd_o_p0ref = wpfloat("20.0")
        rho = random_field(grid, CellDim, KDim, low=1, high=2, dtype=wpfloat)
        theta_v = random_field(grid, CellDim, KDim, low=1, high=2, dtype=wpfloat)
        exner = random_field(grid, CellDim, KDim, low=1, high=2, dtype=wpfloat)

        return dict(
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
