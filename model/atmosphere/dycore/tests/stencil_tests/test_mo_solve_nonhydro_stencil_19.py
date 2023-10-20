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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_19 import (
    mo_solve_nonhydro_stencil_19,
)
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil19(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_19
    OUTPUTS = ("z_gradh_exner",)

    @staticmethod
    def reference(
        mesh,
        inv_dual_edge_length: np.array,
        z_exner_ex_pr: np.array,
        ddxn_z_full: np.array,
        c_lin_e: np.array,
        z_dexner_dz_c_1: np.array,
        **kwargs,
    ) -> dict:
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
        c_lin_e = np.expand_dims(c_lin_e, axis=-1)

        z_exner_ex_pr_e2c = z_exner_ex_pr[mesh.e2c]
        z_exner_ex_weighted = z_exner_ex_pr_e2c[:, 1] - z_exner_ex_pr_e2c[:, 0]

        z_gradh_exner = inv_dual_edge_length * z_exner_ex_weighted - ddxn_z_full * np.sum(
            c_lin_e * z_dexner_dz_c_1[mesh.e2c], axis=1
        )
        return dict(z_gradh_exner=z_gradh_exner)

    @pytest.fixture
    def input_data(self, mesh):
        inv_dual_edge_length = random_field(mesh, EdgeDim, dtype=wpfloat)
        z_exner_ex_pr = random_field(mesh, CellDim, KDim, dtype=vpfloat)
        ddxn_z_full = random_field(mesh, EdgeDim, KDim, dtype=vpfloat)
        c_lin_e = random_field(mesh, EdgeDim, E2CDim, dtype=wpfloat)
        z_dexner_dz_c_1 = random_field(mesh, CellDim, KDim, dtype=vpfloat)
        z_gradh_exner = random_field(mesh, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_gradh_exner=z_gradh_exner,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_edges),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
