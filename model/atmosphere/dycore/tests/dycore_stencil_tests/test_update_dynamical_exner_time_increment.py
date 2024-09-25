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

from icon4py.model.atmosphere.dycore.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestUpdateDynamicalExnerTimeIncrement(StencilTest):
    PROGRAM = update_dynamical_exner_time_increment
    OUTPUTS = ("exner_dyn_incr_lastsubstep",)

    @staticmethod
    def reference(
        grid,
        exner: np.array,
        ddt_exner_phy: np.array,
        exner_dyn_incr: np.array,
        ndyn_substeps_var: float,
        dtime: float,
        **kwargs,
    ) -> dict:
        exner_dyn_incr_lastsubstep = exner - (
            exner_dyn_incr + ndyn_substeps_var * dtime * ddt_exner_phy
        )
        return dict(exner_dyn_incr_lastsubstep=exner_dyn_incr_lastsubstep)

    @pytest.fixture
    def input_data(self, grid):
        ndyn_substeps_var, dtime = wpfloat("10.0"), wpfloat("12.0")
        exner = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddt_exner_phy = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        exner_dyn_incr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        exner_dyn_incr_lastsubstep = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            exner=exner,
            ddt_exner_phy=ddt_exner_phy,
            exner_dyn_incr=exner_dyn_incr,
            exner_dyn_incr_lastsubstep=exner_dyn_incr_lastsubstep,
            ndyn_substeps_var=ndyn_substeps_var,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
