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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_17 import (
    mo_solve_nonhydro_stencil_17,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil17(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_17
    OUTPUTS = ("z_graddiv_vn",)

    @staticmethod
    def reference(
        mesh,
        hmask_dd3d: np.array,
        scalfac_dd3d: np.array,
        inv_dual_edge_length: np.array,
        z_dwdz_dd: np.array,
        z_graddiv_vn: np.array,
        **kwargs,
    ) -> dict:
        scalfac_dd3d = np.expand_dims(scalfac_dd3d, axis=0)
        hmask_dd3d = np.expand_dims(hmask_dd3d, axis=-1)
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        z_dwdz_dd_e2c = z_dwdz_dd[mesh.e2c]
        z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1] - z_dwdz_dd_e2c[:, 0]

        z_graddiv_vn = z_graddiv_vn + (
            hmask_dd3d * scalfac_dd3d * inv_dual_edge_length * z_dwdz_dd_weighted
        )
        return dict(z_graddiv_vn=z_graddiv_vn)

    @pytest.fixture
    def input_data(self, mesh):
        hmask_dd3d = random_field(mesh, EdgeDim, dtype=wpfloat)
        scalfac_dd3d = random_field(mesh, KDim, dtype=wpfloat)
        inv_dual_edge_length = random_field(mesh, EdgeDim, dtype=wpfloat)
        z_dwdz_dd = random_field(mesh, CellDim, KDim, dtype=vpfloat)
        z_graddiv_vn = random_field(mesh, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_edges),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
