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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoVelocityAdvectionStencil02VnIe(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_02
    OUTPUTS = ("vn_ie", "z_kin_hor_e")

    @staticmethod
    def mo_velocity_advection_stencil_02_vn_ie_numpy(wgtfac_e: np.array, vn: np.array) -> np.array:
        vn_ie_k_minus_1 = np.roll(vn, shift=1, axis=1)
        vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_ie_k_minus_1
        vn_ie[:, 0] = 0
        return vn_ie

    @staticmethod
    def mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(vn: np.array, vt: np.array) -> np.array:
        z_kin_hor_e = 0.5 * (vn * vn + vt * vt)
        z_kin_hor_e[:, 0] = 0
        return z_kin_hor_e

    @classmethod
    def reference(cls, mesh, wgtfac_e: np.array, vn: np.array, vt: np.array, **kwargs) -> dict:
        vn_ie = cls.mo_velocity_advection_stencil_02_vn_ie_numpy(wgtfac_e, vn)
        z_kin_hor_e = cls.mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(vn, vt)
        return dict(
            vn_ie=vn_ie[int32(1) : int32(mesh.n_cells), int32(1) : int32(mesh.k_level)],
            z_kin_hor_e=z_kin_hor_e[int32(1) : int32(mesh.n_cells), int32(1) : int32(mesh.k_level)],
        )

    @pytest.fixture
    def input_data(self, mesh):
        wgtfac_e = random_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        vt = random_field(mesh, EdgeDim, KDim)

        vn_ie = zero_field(mesh, EdgeDim, KDim)
        z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)

        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie[int32(1) : int32(mesh.n_cells), int32(1) : int32(mesh.k_level)],
            z_kin_hor_e=z_kin_hor_e[int32(1) : int32(mesh.n_cells), int32(1) : int32(mesh.k_level)],
            horizontal_start=int32(1),
            horizontal_end=int32(mesh.n_cells),
            vertical_start=int32(1),
            vertical_end=int32(mesh.k_level),
        )
