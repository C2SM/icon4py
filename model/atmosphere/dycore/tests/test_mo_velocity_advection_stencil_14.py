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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_14 import (
    mo_velocity_advection_stencil_14,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import (
    random_field,
    random_mask,
    zero_field,
    StencilTest
)



class TestMoVelocityAdvectionStencil14(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_14
    OUTPUTS = ("cfl_clipping", "pre_levelmask", "vcfl", "z_w_con_c")

    @staticmethod
    def reference(
        mesh, ddqz_z_half: np.array, z_w_con_c: np.array, cfl_w_limit, dtime, **kwargs
    ) -> dict:
        num_rows, num_cols = z_w_con_c.shape
        cfl_clipping = np.where(
            np.abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
            np.ones([num_rows, num_cols]),
            np.zeros_like(z_w_con_c),
        )
        num_rows, num_cols = cfl_clipping.shape
        pre_levelmask = np.where(
            cfl_clipping == 1.0,
            np.ones([num_rows, num_cols]),
            np.zeros_like(cfl_clipping),
        )
        vcfl = np.where(cfl_clipping == 1.0, z_w_con_c * dtime / ddqz_z_half, 0.0)
        z_w_con_c = np.where(
            (cfl_clipping == 1.0) & (vcfl < -0.85),
            -0.85 * ddqz_z_half / dtime,
            z_w_con_c,
        )
        z_w_con_c = np.where(
            (cfl_clipping == 1.0) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c
        )

        return dict(
            cfl_clipping=cfl_clipping,
            pre_levelmask=pre_levelmask,
            vcfl=vcfl,
            z_w_con_c=z_w_con_c,
        )

    @pytest.fixture
    def input_data(self, mesh):
        ddqz_z_half = random_field(mesh, CellDim, KDim)
        z_w_con_c = random_field(mesh, CellDim, KDim)
        cfl_clipping = random_mask(mesh, CellDim, KDim, dtype=bool)
        pre_levelmask = random_mask(
            mesh, CellDim, KDim, dtype=bool
        )  # TODO should be just a K field
        vcfl = zero_field(mesh, CellDim, KDim)
        cfl_w_limit = 5.0
        dtime = 9.0

        return dict(
            ddqz_z_half=ddqz_z_half,
            z_w_con_c=z_w_con_c,
            cfl_clipping=cfl_clipping,
            pre_levelmask=pre_levelmask,
            vcfl=vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        )
