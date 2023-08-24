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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_58 import (
    mo_solve_nonhydro_stencil_58,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil58(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_58
    OUTPUTS = ("mass_flx_ic",)

    @staticmethod
    def reference(
        mesh,
        z_contr_w_fl_l: np.array,
        rho_ic: np.array,
        vwind_impl_wgt: np.array,
        w: np.array,
        mass_flx_ic: np.array,
        r_nsubsteps,
        **kwargs,
    ) -> dict:
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
        mass_flx_ic = mass_flx_ic + (
            r_nsubsteps * (z_contr_w_fl_l + rho_ic * vwind_impl_wgt * w)
        )
        return dict(mass_flx_ic=mass_flx_ic)

    @pytest.fixture
    def input_data(self, mesh):
        z_contr_w_fl_l = random_field(mesh, CellDim, KDim)
        rho_ic = random_field(mesh, CellDim, KDim)
        vwind_impl_wgt = random_field(mesh, CellDim)
        w = random_field(mesh, CellDim, KDim)
        mass_flx_ic = random_field(mesh, CellDim, KDim)
        r_nsubsteps = 7.0

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
        )
