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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_56_63 import (
    mo_solve_nonhydro_stencil_56_63,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field


class TestMoSolveNonhydroStencil5663(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_56_63
    OUTPUTS = ("z_dwdz_dd",)

    @staticmethod
    def reference(
        mesh, inv_ddqz_z_full: np.array, w: np.array, w_concorr_c: np.array, **kwargs
    ) -> np.array:
        z_dwdz_dd = inv_ddqz_z_full * (
            (w[:, :-1] - w[:, 1:]) - (w_concorr_c[:, :-1] - w_concorr_c[:, 1:])
        )
        return dict(z_dwdz_dd=z_dwdz_dd)

    @pytest.fixture
    def input_data(self, mesh):
        inv_ddqz_z_full = random_field(mesh, CellDim, KDim)
        w = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        w_concorr_c = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        z_dwdz_dd = random_field(mesh, CellDim, KDim)

        return dict(
            inv_ddqz_z_full=inv_ddqz_z_full,
            w=w,
            w_concorr_c=w_concorr_c,
            z_dwdz_dd=z_dwdz_dd,
        )
