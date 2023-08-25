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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_39 import (
    mo_solve_nonhydro_stencil_39,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    zero_field,
)


class TestMoSolveNonhydroStencil39(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_39
    OUTPUTS = ("w_concorr_c",)

    @staticmethod
    def reference(
        mesh,
        e_bln_c_s: np.array,
        z_w_concorr_me: np.array,
        wgtfac_c: np.array,
        **kwargs,
    ) -> np.array:
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        z_w_concorr_me_offset_1 = np.roll(z_w_concorr_me, shift=1, axis=1)
        z_w_concorr_mc_m0 = np.sum(e_bln_c_s * z_w_concorr_me[mesh.c2e], axis=1)
        z_w_concorr_mc_m1 = np.sum(
            e_bln_c_s * z_w_concorr_me_offset_1[mesh.c2e], axis=1
        )
        w_concorr_c = (
            wgtfac_c * z_w_concorr_mc_m0 + (1.0 - wgtfac_c) * z_w_concorr_mc_m1
        )
        return dict(w_concorr_c=w_concorr_c)

    @pytest.fixture
    def input_data(self, mesh):
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
        wgtfac_c = random_field(mesh, CellDim, KDim)
        w_concorr_c = zero_field(mesh, CellDim, KDim)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w_concorr_c=w_concorr_c,
        )
