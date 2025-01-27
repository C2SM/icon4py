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

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


class TestAddAnalysisIncrementsFromDataAssimilation(StencilTest):
    PROGRAM = add_analysis_increments_from_data_assimilation
    OUTPUTS = ("z_rho_expl", "z_exner_expl")

    @staticmethod
    def reference(
        grid,
        z_rho_expl: np.array,
        rho_incr: np.array,
        z_exner_expl: np.array,
        exner_incr: np.array,
        iau_wgt_dyn,
        **kwargs,
    ) -> dict:
        z_rho_expl = z_rho_expl + iau_wgt_dyn * rho_incr
        z_exner_expl = z_exner_expl + iau_wgt_dyn * exner_incr
        return dict(z_rho_expl=z_rho_expl, z_exner_expl=z_exner_expl)

    @pytest.fixture
    def input_data(self, grid):
        z_exner_expl = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_incr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rho_expl = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        rho_incr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        iau_wgt_dyn = wpfloat("8.0")

        return dict(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
