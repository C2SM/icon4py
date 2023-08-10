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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_38 import (
    mo_solve_nonhydro_stencil_38,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoSolveNonhydroStencil38(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_38
    OUTPUTS = ("vn_ie",)

    @staticmethod
    def reference(mesh, vn: np.array, wgtfacq_e: np.array, **kwargs) -> np.array:
        vn_ie = (
            np.roll(wgtfacq_e, shift=1, axis=1) * np.roll(vn, shift=1, axis=1)
            + np.roll(wgtfacq_e, shift=2, axis=1) * np.roll(vn, shift=2, axis=1)
            + np.roll(wgtfacq_e, shift=3, axis=1) * np.roll(vn, shift=3, axis=1)
        )
        return dict(vn_ie=vn_ie)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfacq_e = zero_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        vn_ie = zero_field(mesh, EdgeDim, KDim)

        return dict(
            vn=vn,
            wgtfacq_e=wgtfacq_e,
            vn_ie=vn_ie,
        )
