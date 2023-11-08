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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_19 import (
    mo_velocity_advection_stencil_19,
)
from icon4py.model.common.dimension import CellDim, E2CDim, E2VDim, ECDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)


class TestMoVelocityAdvectionStencil19(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_19
    OUTPUTS = ("ddt_vn_apc",)

    @staticmethod
    def reference(
        grid,
        z_kin_hor_e: np.array,
        coeff_gradekin: np.array,
        z_ekinh: np.array,
        zeta: np.array,
        vt: np.array,
        f_e: np.array,
        c_lin_e: np.array,
        z_w_con_c_full: np.array,
        vn_ie: np.array,
        ddqz_z_full_e: np.array,
        **kwargs,
    ) -> np.array:
        e2c = grid.connectivities[E2CDim]
        z_ekinh_e2c = z_ekinh[e2c]
        coeff_gradekin = coeff_gradekin.reshape(e2c.shape)
        coeff_gradekin = np.expand_dims(coeff_gradekin, axis=-1)
        f_e = np.expand_dims(f_e, axis=-1)
        c_lin_e = np.expand_dims(c_lin_e, axis=-1)

        ddt_vn_apc = -(
            (coeff_gradekin[:, 0] - coeff_gradekin[:, 1]) * z_kin_hor_e
            + (-coeff_gradekin[:, 0] * z_ekinh_e2c[:, 0] + coeff_gradekin[:, 1] * z_ekinh_e2c[:, 1])
            + vt * (f_e + 0.5 * np.sum(zeta[grid.connectivities[E2VDim]], axis=1))
            + np.sum(z_w_con_c_full[e2c] * c_lin_e, axis=1)
            * (vn_ie[:, :-1] - vn_ie[:, 1:])
            / ddqz_z_full_e
        )
        return dict(ddt_vn_apc=ddt_vn_apc)

    @pytest.fixture
    def input_data(self, grid):
        z_kin_hor_e = random_field(grid, EdgeDim, KDim)
        coeff_gradekin = random_field(grid, EdgeDim, E2CDim)
        coeff_gradekin_new = as_1D_sparse_field(coeff_gradekin, ECDim)
        z_ekinh = random_field(grid, CellDim, KDim)
        zeta = random_field(grid, VertexDim, KDim)
        vt = random_field(grid, EdgeDim, KDim)
        f_e = random_field(grid, EdgeDim)
        c_lin_e = random_field(grid, EdgeDim, E2CDim)
        z_w_con_c_full = random_field(grid, CellDim, KDim)
        vn_ie = random_field(grid, EdgeDim, KDim, extend={KDim: 1})
        ddqz_z_full_e = random_field(grid, EdgeDim, KDim)
        ddt_vn_apc = zero_field(grid, EdgeDim, KDim)

        return dict(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=coeff_gradekin_new,
            z_ekinh=z_ekinh,
            zeta=zeta,
            vt=vt,
            f_e=f_e,
            c_lin_e=c_lin_e,
            vn_ie=vn_ie,
            z_w_con_c_full=z_w_con_c_full,
            ddqz_z_full_e=ddqz_z_full_e,
            ddt_vn_apc=ddt_vn_apc,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
