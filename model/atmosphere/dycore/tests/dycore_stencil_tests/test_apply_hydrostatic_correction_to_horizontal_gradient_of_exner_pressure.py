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

from icon4py.model.atmosphere.dycore.stencils.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask
from icon4py.model.common.type_alias import vpfloat


class TestApplyHydrostaticCorrectionToHorizontalGradientOfExnerPressure(StencilTest):
    PROGRAM = apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        grid,
        ipeidx_dsl: np.array,
        pg_exdist: np.array,
        z_hydro_corr: np.array,
        z_gradh_exner: np.array,
        **kwargs,
    ) -> dict:
        z_hydro_corr = np.expand_dims(z_hydro_corr, axis=-1)
        z_gradh_exner = np.where(
            ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner
        )
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, grid):
        ipeidx_dsl = random_mask(grid, EdgeDim, KDim)
        pg_exdist = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_hydro_corr = random_field(grid, EdgeDim, dtype=vpfloat)
        z_gradh_exner = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            z_hydro_corr=z_hydro_corr,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
