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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_04 import (
    mo_solve_nonhydro_stencil_04,
)
from icon4py.common.dimension import CellDim, KDim

from .conftest import StencilTest
from .test_utils.helpers import random_field, zero_field


class TestMoSolveNonhydroStencil04(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_04
    OUTPUTS = ("z_exner_ic",)

    @staticmethod
    def reference(
        mesh,
        z_exner_ex_pr: np.array,
        wgtfacq_c: np.array,
        z_exner_ic: np.array,
    ) -> np.array:
        z_exner_ic[:, 3:] = (
            np.roll(wgtfacq_c, shift=1, axis=1)
            * np.roll(z_exner_ex_pr, shift=1, axis=1)
            + np.roll(wgtfacq_c, shift=2, axis=1)
            * np.roll(z_exner_ex_pr, shift=2, axis=1)
            + np.roll(wgtfacq_c, shift=3, axis=1)
            * np.roll(z_exner_ex_pr, shift=3, axis=1)
        )[:, 3:]
        return {"z_exner_ic": z_exner_ic}

    @pytest.fixture
    def input_data(self, mesh):
        z_exner_ex_pr = random_field(mesh, CellDim, KDim)
        wgtfacq_c = random_field(mesh, CellDim, KDim)
        z_exner_ic = zero_field(mesh, CellDim, KDim)

        return dict(
            z_exner_ex_pr=z_exner_ex_pr,
            wgtfacq_c=wgtfacq_c,
            z_exner_ic=z_exner_ic,
        )
