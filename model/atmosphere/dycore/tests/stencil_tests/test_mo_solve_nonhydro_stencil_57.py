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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_57 import (
    mo_solve_nonhydro_stencil_57,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, zero_field


class TestMoSolveNonhydroStencil57(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_57
    OUTPUTS = ("mass_flx_ic",)

    @staticmethod
    def reference(mesh, mass_flx_ic: np.array, **kwargs) -> dict:
        mass_flx_ic = np.zeros_like(mass_flx_ic)
        return dict(mass_flx_ic=mass_flx_ic)

    @pytest.fixture
    def input_data(self, mesh):
        mass_flx_ic = zero_field(mesh, CellDim, KDim)

        return dict(
            mass_flx_ic=mass_flx_ic,
        )
