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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_54 import (
    mo_solve_nonhydro_stencil_54,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


def mo_solve_nonhydro_stencil_54_numpy(
    mesh, z_raylfac: np.array, w_1: np.array, w: np.array
) -> np.array:
    z_raylfac = np.expand_dims(z_raylfac, axis=0)
    # w_1 = np.expand_dims(w_1, axis=-1)
    w = z_raylfac * w + (1.0 - z_raylfac) * w_1
    return w


class TestMoSolveNonhydroStencil54(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_54
    OUTPUTS = ("w",)

    @staticmethod
    def reference(mesh, z_raylfac: np.array, w_1: np.array, w: np.array, **kwargs) -> dict:
        w = mo_solve_nonhydro_stencil_54_numpy(mesh, z_raylfac, w_1, w)
        return dict(w=w)

    @pytest.fixture
    def input_data(self, mesh):
        z_raylfac = random_field(mesh, KDim)
        w_1 = random_field(mesh, CellDim, KDim)
        w = random_field(mesh, CellDim, KDim)

        return dict(
            z_raylfac=z_raylfac,
            w_1=w_1,
            w=w,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_cells),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
