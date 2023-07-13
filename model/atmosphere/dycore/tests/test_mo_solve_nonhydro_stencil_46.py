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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_46 import (
    mo_solve_nonhydro_stencil_46,
)
from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.common.test_utils.helpers import zero_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestMoSolveNonhydroStencil46(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_46
    OUTPUTS = ("w_nnew", "z_contr_w_fl_l")

    @staticmethod
    def reference(mesh, w_nnew: np.array, z_contr_w_fl_l: np.array, **kwargs) -> dict:
        w_nnew = np.zeros_like(w_nnew)
        z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
        return dict(w_nnew=w_nnew, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, mesh):
        z_contr_w_fl_l = zero_field(mesh, CellDim, KDim)
        w_nnew = zero_field(mesh, CellDim, KDim)

        return dict(
            w_nnew=w_nnew,
            z_contr_w_fl_l=z_contr_w_fl_l,
        )
