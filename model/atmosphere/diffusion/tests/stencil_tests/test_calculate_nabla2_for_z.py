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
from gt4py.next.program_processors.otf_compile_executor import OTFCompileExecutor

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    calculate_nabla2_for_z,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestCalculateNabla2ForZ(StencilTest):
    PROGRAM = calculate_nabla2_for_z
    OUTPUTS = ("z_nabla2_e",)

    @staticmethod
    def reference(
        grid,
        kh_smag_e: np.array,
        inv_dual_edge_length: np.array,
        theta_v: np.array,
        **kwargs,
    ) -> np.array:
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        theta_v_e2c = theta_v[grid.connectivities[E2CDim]]
        theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

        z_nabla2_e = kh_smag_e * inv_dual_edge_length * theta_v_weighted
        return dict(z_nabla2_e=z_nabla2_e)

    @pytest.fixture
    def input_data(self, grid, backend):
        if isinstance(backend, OTFCompileExecutor):
            pytest.skip(
                "Execution domain needs to be restricted or boundary taken into account in stencil."
            )
        kh_smag_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        inv_dual_edge_length = random_field(grid, EdgeDim, dtype=wpfloat)
        theta_v = random_field(grid, CellDim, KDim, dtype=wpfloat)
        z_nabla2_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v,
            z_nabla2_e=z_nabla2_e,
        )
