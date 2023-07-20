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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_31 import (
    mo_solve_nonhydro_stencil_31,
)
from icon4py.model.common.dimension import E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoSolveNonhydroStencil31(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_31
    OUTPUTS = ("z_vn_avg",)

    @staticmethod
    def reference(mesh, e_flx_avg: np.array, vn: np.array, **kwargs) -> np.array:
        geofac_grdiv = np.expand_dims(e_flx_avg, axis=-1)
        z_vn_avg = np.sum(vn[mesh.e2c2eO] * geofac_grdiv, axis=1)
        return dict(z_vn_avg=z_vn_avg)

    @pytest.fixture
    def input_data(self, mesh):
        e_flx_avg = random_field(mesh, EdgeDim, E2C2EODim)
        vn = random_field(mesh, EdgeDim, KDim)
        z_vn_avg = zero_field(mesh, EdgeDim, KDim)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            z_vn_avg=z_vn_avg,
        )
