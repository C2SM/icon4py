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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_25 import (
    mo_solve_nonhydro_stencil_25,
)
from icon4py.model.common.dimension import E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoSolveNonhydroStencil25(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_25
    OUTPUTS = ("z_graddiv2_vn",)

    @staticmethod
    def reference(
        mesh, geofac_grdiv: np.array, z_graddiv_vn: np.array, **kwargs
    ) -> np.array:
        geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
        z_graddiv2_vn = np.sum(z_graddiv_vn[mesh.e2c2eO] * geofac_grdiv, axis=1)
        return dict(z_graddiv2_vn=z_graddiv2_vn)

    @pytest.fixture
    def input_data(self, mesh):
        z_graddiv_vn = random_field(mesh, EdgeDim, KDim)
        geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
        z_graddiv2_vn = zero_field(mesh, EdgeDim, KDim)

        return dict(
            geofac_grdiv=geofac_grdiv,
            z_graddiv_vn=z_graddiv_vn,
            z_graddiv2_vn=z_graddiv2_vn,
        )
