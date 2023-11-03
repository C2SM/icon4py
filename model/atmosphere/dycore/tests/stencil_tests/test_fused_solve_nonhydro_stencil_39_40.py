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

from icon4py.model.atmosphere.dycore.fused_solve_nonhydro_stencil_39_40 import (
    fused_solve_nonhydro_stencil_39_40,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat

from .test_mo_solve_nonhydro_stencil_39 import mo_solve_nonhydro_stencil_39_numpy
from .test_mo_solve_nonhydro_stencil_40 import mo_solve_nonhydro_stencil_40_numpy


def _fused_solve_nonhydro_stencil_39_40_numpy(
    mesh, e_bln_c_s, z_w_concorr_me, wgtfac_c, wgtfacq_c, vert_idx, nlev, nflatlev
):
    w_concorr_c = np.where(
        (nflatlev < vert_idx) & (vert_idx < nlev),
        mo_solve_nonhydro_stencil_39_numpy(mesh, e_bln_c_s, z_w_concorr_me, wgtfac_c),
        mo_solve_nonhydro_stencil_40_numpy(mesh, e_bln_c_s, z_w_concorr_me, wgtfacq_c),
    )

    w_concorr_c_res = np.zeros_like(w_concorr_c)
    w_concorr_c_res[:, -1] = w_concorr_c[:, -1]
    return w_concorr_c_res


class TestFusedSolveNonhydroStencil39To40(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_39_40
    OUTPUTS = ("w_concorr_c",)

    @staticmethod
    def reference(
        mesh,
        e_bln_c_s: np.array,
        z_w_concorr_me: np.array,
        wgtfac_c: np.array,
        wgtfacq_c: np.array,
        vert_idx: np.array,
        nlev: int,
        nflatlev: int,
        **kwargs,
    ) -> dict:
        w_concorr_c_result = _fused_solve_nonhydro_stencil_39_40_numpy(
            mesh, e_bln_c_s, z_w_concorr_me, wgtfac_c, wgtfacq_c, vert_idx, nlev, nflatlev
        )
        return dict(w_concorr_c=w_concorr_c_result)

    @pytest.fixture
    def input_data(self, mesh):
        e_bln_c_s = random_field(mesh, CEDim, dtype=wpfloat)
        z_w_concorr_me = random_field(mesh, EdgeDim, KDim, dtype=vpfloat)
        wgtfac_c = random_field(mesh, CellDim, KDim, dtype=vpfloat)
        wgtfacq_c = random_field(mesh, CellDim, KDim, dtype=vpfloat)
        w_concorr_c = zero_field(mesh, CellDim, KDim, dtype=vpfloat)

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        nlev = mesh.k_level
        nflatlev = 13

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            vert_idx=vert_idx,
            nlev=nlev,
            nflatlev=nflatlev,
            w_concorr_c=w_concorr_c,
        )
