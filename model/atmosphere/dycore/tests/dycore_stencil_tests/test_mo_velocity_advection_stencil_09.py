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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_09 import (
    mo_velocity_advection_stencil_09,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)


def mo_velocity_advection_stencil_09_numpy(
    mesh, z_w_concorr_me: np.array, e_bln_c_s: np.array
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_mc = np.sum(z_w_concorr_me[mesh.c2e] * e_bln_c_s, axis=1)
    return z_w_concorr_mc


class TestMoVelocityAdvectionStencil09(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_09
    OUTPUTS = ("z_w_concorr_mc",)

    @staticmethod
    def reference(mesh, z_w_concorr_me: np.array, e_bln_c_s: np.array, **kwargs) -> dict:
        z_w_concorr_mc = mo_velocity_advection_stencil_09_numpy(mesh, z_w_concorr_me, e_bln_c_s)
        return dict(z_w_concorr_mc=z_w_concorr_mc)

    @pytest.fixture
    def input_data(self, mesh):
        z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        z_w_concorr_mc = zero_field(mesh, CellDim, KDim)

        return dict(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
            z_w_concorr_mc=z_w_concorr_mc,
        )
