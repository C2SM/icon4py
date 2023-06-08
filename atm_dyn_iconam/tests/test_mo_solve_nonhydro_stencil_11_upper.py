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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_11_upper import (
    mo_solve_nonhydro_stencil_11_upper,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field, zero_field


class TestMoSolveNonhydroStencil11Upper(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_11_upper
    OUTPUTS = ("z_theta_v_pr_ic", "theta_v_ic")

    @staticmethod
    def reference(
        mesh,
        wgtfacq_c: np.array,
        z_rth_pr: np.array,
        theta_ref_ic: np.array,
        z_theta_v_pr_ic: np.array,
        **kwargs,
    ) -> tuple[np.array, np.array]:
        z_theta_v_pr_ic = (
            np.roll(wgtfacq_c, shift=1, axis=1) * np.roll(z_rth_pr, shift=1, axis=1)
            + np.roll(wgtfacq_c, shift=2, axis=1) * np.roll(z_rth_pr, shift=2, axis=1)
            + np.roll(wgtfacq_c, shift=3, axis=1) * np.roll(z_rth_pr, shift=3, axis=1)
        )
        theta_v_ic = theta_ref_ic + z_theta_v_pr_ic
        return dict(z_theta_v_pr_ic=z_theta_v_pr_ic, theta_v_ic=theta_v_ic)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfacq_c = random_field(mesh, CellDim, KDim)
        z_rth_pr = random_field(mesh, CellDim, KDim)
        theta_ref_ic = random_field(mesh, CellDim, KDim)
        z_theta_v_pr_ic = random_field(mesh, CellDim, KDim)
        theta_v_ic = zero_field(mesh, CellDim, KDim)

        return dict(
            wgtfacq_c=wgtfacq_c,
            z_rth_pr=z_rth_pr,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
        )
