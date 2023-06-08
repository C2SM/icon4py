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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.common.dimension import EdgeDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field


class TestMoSolveNonhydro4thOrderDivdamp(StencilTest):
    PROGRAM = mo_solve_nonhydro_4th_order_divdamp
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        mesh,
        scal_divdamp: np.array,
        z_graddiv2_vn: np.array,
        vn: np.array,
    ) -> np.array:
        scal_divdamp = np.expand_dims(scal_divdamp, axis=0)
        vn = vn + (scal_divdamp * z_graddiv2_vn)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, mesh):
        scal_divdamp = random_field(mesh, KDim)
        z_graddiv2_vn = random_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)

        return dict(
            scal_divdamp=scal_divdamp,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
        )
