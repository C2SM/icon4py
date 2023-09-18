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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_01 import (
    mo_velocity_advection_stencil_01,
)
from icon4py.model.common.dimension import E2C2EDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoVelocityAdvectionStencil01(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_01
    OUTPUTS = ("vt",)

    @staticmethod
    def reference(mesh, vn: np.array, rbf_vec_coeff_e: np.array, **kwargs) -> np.array:
        rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
        vt = np.sum(vn[mesh.e2c2e] * rbf_vec_coeff_e, axis=1)
        return dict(vt=vt)

    @pytest.fixture
    def input_data(self, mesh):
        vn = random_field(mesh, EdgeDim, KDim)
        rbf_vec_coeff_e = random_field(mesh, EdgeDim, E2C2EDim)
        vt = zero_field(mesh, EdgeDim, KDim)

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            vt=vt,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_edges),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
