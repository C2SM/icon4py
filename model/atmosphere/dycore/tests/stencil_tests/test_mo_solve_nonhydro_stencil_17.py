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
from icon4py.model.common.dimension import CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


def mo_solve_nonhydro_stencil_17_numpy(
    grid,
    hmask_dd3d: np.array,
    scalfac_dd3d: np.array,
    inv_dual_edge_length: np.array,
    z_dwdz_dd: np.array,
    z_graddiv_vn: np.array,
) -> np.array:
    scalfac_dd3d = np.expand_dims(scalfac_dd3d, axis=0)
    hmask_dd3d = np.expand_dims(hmask_dd3d, axis=-1)
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    z_dwdz_dd_e2c = z_dwdz_dd[grid.connectivities[E2CDim]]
    z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1] - z_dwdz_dd_e2c[:, 0]

    z_graddiv_vn = z_graddiv_vn + (
        hmask_dd3d * scalfac_dd3d * inv_dual_edge_length * z_dwdz_dd_weighted
    )
    return z_graddiv_vn


class TestMoSolveNonhydroStencil17(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_17
    OUTPUTS = ("z_graddiv_vn",)

    @staticmethod
    def reference(
        grid,
        hmask_dd3d: np.array,
        scalfac_dd3d: np.array,
        inv_dual_edge_length: np.array,
        z_dwdz_dd: np.array,
        z_graddiv_vn: np.array,
        **kwargs,
    ) -> dict:
        z_graddiv_vn = mo_solve_nonhydro_stencil_17_numpy(
            grid,
            hmask_dd3d,
            scalfac_dd3d,
            inv_dual_edge_length,
            z_dwdz_dd,
            z_graddiv_vn,
        )
        return dict(z_graddiv_vn=z_graddiv_vn)

    @pytest.fixture
    def input_data(self, grid):
        if np.any(grid.connectivities[E2CDim] == -1):
            pytest.xfail("Stencil does not support missing neighbors.")

        hmask_dd3d = random_field(grid, EdgeDim)
        scalfac_dd3d = random_field(grid, KDim)
        inv_dual_edge_length = random_field(grid, EdgeDim)
        z_dwdz_dd = random_field(grid, CellDim, KDim)
        z_graddiv_vn = random_field(grid, EdgeDim, KDim)

        return dict(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
