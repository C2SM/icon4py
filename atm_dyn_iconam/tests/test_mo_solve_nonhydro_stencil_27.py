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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_27 import (
    mo_solve_nonhydro_stencil_27,
)
from icon4py.common.dimension import EdgeDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field


class TestMoSolveNonhydroStencil27(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_27
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        mesh,
        scal_divdamp: np.array,
        bdy_divdamp: np.array,
        nudgecoeff_e: np.array,
        z_graddiv2_vn: np.array,
        vn: np.array,
        **kwargs,
    ) -> dict:
        nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
        vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, mesh):
        scal_divdamp = random_field(mesh, KDim)
        bdy_divdamp = random_field(mesh, KDim)
        nudgecoeff_e = random_field(mesh, EdgeDim)
        z_graddiv2_vn = random_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)

        return dict(
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
        )
