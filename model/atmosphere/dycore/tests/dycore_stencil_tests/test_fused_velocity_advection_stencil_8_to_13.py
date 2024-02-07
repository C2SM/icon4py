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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_8_to_13 import (
    fused_velocity_advection_stencil_8_to_13,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)


class TestFusedVelocityAdvectionStencil8To13(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_8_to_13
    OUTPUTS = (
        "z_ekinh",
        "w_concorr_c",
        "z_w_con_c",
    )

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:

        z_ekinh = 0.0
        w_concorr_c = 0.0
        z_w_con_c = 0.0

        return dict(z_ekinh=z_ekinh, w_concorr_c=w_concorr_c, z_w_con_c=z_w_con_c)

    @pytest.fixture
    def input_data(self, mesh):

        z_kin_hor_e = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        z_ekinh = zero_field(mesh, CellDim, KDim)
        z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
        z_w_concorr_mc = zero_field(mesh, CellDim, KDim)
        wgtfac_c = random_field(mesh, CellDim, KDim)
        w_concorr_c = zero_field(mesh, CellDim, KDim)
        w = random_field(mesh, CellDim, KDim)
        z_w_con_c = zero_field(mesh, CellDim, KDim)

        cfl_w_limit = 5.0
        dtime = 9.0

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        nlev = mesh.k_level
        nflatlev = 13
        nrdmax = 10

        istep = 1
        return dict(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w=w,
            z_w_concorr_mc=z_w_concorr_mc,
            w_concorr_c=w_concorr_c,
            z_ekinh=z_ekinh,
            z_w_con_c=z_w_con_c,
            k=k,
            istep=istep,
            nlev=nlev,
            nflatlev=nflatlev,
        )
