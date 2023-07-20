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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_08 import (
    mo_velocity_advection_stencil_08,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoVelocityAdvectionStencil08(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_08
    OUTPUTS = ("z_ekinh",)

    @staticmethod
    def reference(mesh, z_kin_hor_e: np.array, e_bln_c_s: np.array, **kwargs) -> np.array:
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        z_ekinh = np.sum(z_kin_hor_e[mesh.c2e] * e_bln_c_s, axis=1)
        return dict(z_ekinh=z_ekinh)

    @pytest.fixture
    def input_data(self, mesh):
        z_kin_hor_e = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        z_ekinh = zero_field(mesh, CellDim, KDim)

        return dict(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=e_bln_c_s,
            z_ekinh=z_ekinh,
        )
