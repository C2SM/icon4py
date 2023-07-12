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

from icon4py.model.atm_dyn_iconam.mo_solve_nonhydro_stencil_23 import (
    mo_solve_nonhydro_stencil_23,
)
from icon4py.model.common.dimension import EdgeDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.stencil_test import StencilTest


class TestMoSolveNonhydroStencil23(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_23
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        mesh,
        vn_nnow: np.array,
        ddt_vn_adv_ntl1: np.array,
        ddt_vn_adv_ntl2: np.array,
        ddt_vn_phy: np.array,
        z_theta_v_e: np.array,
        z_gradh_exner: np.array,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        **kwargs,
    ) -> np.array:
        vn_nnew = vn_nnow + dtime * (
            wgt_nnow_vel * ddt_vn_adv_ntl1
            + wgt_nnew_vel * ddt_vn_adv_ntl2
            + ddt_vn_phy
            - cpd * z_theta_v_e * z_gradh_exner
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, mesh):
        vn_nnow = random_field(mesh, EdgeDim, KDim)
        ddt_vn_adv_ntl1 = random_field(mesh, EdgeDim, KDim)
        ddt_vn_adv_ntl2 = random_field(mesh, EdgeDim, KDim)
        ddt_vn_phy = random_field(mesh, EdgeDim, KDim)
        z_theta_v_e = random_field(mesh, EdgeDim, KDim)
        z_gradh_exner = random_field(mesh, EdgeDim, KDim)
        vn_nnew = zero_field(mesh, EdgeDim, KDim)
        dtime = 5.0
        wgt_nnow_vel = 8.0
        wgt_nnew_vel = 7.0
        cpd = 2.0

        return dict(
            vn_nnow=vn_nnow,
            ddt_vn_adv_ntl1=ddt_vn_adv_ntl1,
            ddt_vn_adv_ntl2=ddt_vn_adv_ntl2,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn_nnew=vn_nnew,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
        )
