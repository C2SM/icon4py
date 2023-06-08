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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field


class TestMoSolveNonhydroStencil54(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_54
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        mesh, z_raylfac: np.array, w_1: np.array, w: np.array, **kwargs
    ) -> np.array:
        z_raylfac = np.expand_dims(z_raylfac, axis=0)
        w_1 = np.expand_dims(w_1, axis=-1)
        w = z_raylfac * w + (1.0 - z_raylfac) * w_1
        return dict(w=w)

    @pytest.fixture
    def input_data(self, mesh):
        z_raylfac = random_field(mesh, KDim)
        w_1 = random_field(mesh, CellDim)
        w = random_field(mesh, CellDim, KDim)

        return dict(
            z_raylfac=z_raylfac,
            w_1=w_1,
            w=w,
        )
