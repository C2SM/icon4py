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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_18 import (
    mo_solve_nonhydro_stencil_18,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil18(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_18
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        mesh, inv_dual_edge_length: np.array, z_exner_ex_pr: np.array, **kwargs
    ) -> np.array:
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        z_exner_ex_pr_e2c = z_exner_ex_pr[mesh.e2c]
        z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

        z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, mesh):
        if np.any(mesh.e2c == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        inv_dual_edge_length = random_field(mesh, EdgeDim)
        z_exner_ex_pr = random_field(mesh, CellDim, KDim)
        z_gradh_exner = random_field(mesh, EdgeDim, KDim)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            z_gradh_exner=z_gradh_exner,
        )
