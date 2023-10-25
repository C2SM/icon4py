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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_47 import (
    mo_solve_nonhydro_stencil_47,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil47(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_47
    OUTPUTS = ("w_nnew", "z_contr_w_fl_l")

    @staticmethod
    def reference(grid, w_concorr_c: np.array, z_contr_w_fl_l: np.array, **kwargs) -> dict:
        w_nnew = w_concorr_c
        z_contr_w_fl_l = np.zeros_like(z_contr_w_fl_l)
        return dict(w_nnew=w_nnew, z_contr_w_fl_l=z_contr_w_fl_l)

    @pytest.fixture
    def input_data(self, grid):
        w_concorr_c = random_field(grid, CellDim, KDim)
        z_contr_w_fl_l = zero_field(grid, CellDim, KDim)
        w_nnew = zero_field(grid, CellDim, KDim)

        return dict(
            w_nnew=w_nnew,
            z_contr_w_fl_l=z_contr_w_fl_l,
            w_concorr_c=w_concorr_c,
        )
