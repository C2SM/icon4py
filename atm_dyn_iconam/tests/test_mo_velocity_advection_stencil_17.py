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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_17 import (
    mo_velocity_advection_stencil_17,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim

from .test_utils.helpers import random_field
from .test_utils.stencil import StencilTest


class TestMoVelocityAdvectionStencil17(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_17
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        mesh, e_bln_c_s: np.array, z_v_grad_w: np.array, ddt_w_adv: np.array, **kwargs
    ) -> np.array:
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        ddt_w_adv = ddt_w_adv + np.sum(z_v_grad_w[mesh.c2e] * e_bln_c_s, axis=1)
        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, mesh):
        z_v_grad_w = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        ddt_w_adv = random_field(mesh, CellDim, KDim)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_v_grad_w=z_v_grad_w,
            ddt_w_adv=ddt_w_adv,
        )
