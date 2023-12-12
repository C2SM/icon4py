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

from gt4py.next import common
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next import constructors
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def mo_velocity_advection_stencil_02_vn_ie_numpy(wgtfac_e: np.array, vn: np.array) -> np.array:
    vn_ie_k_minus_1 = np.roll(vn, shift=1, axis=1)
    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_ie_k_minus_1
    vn_ie[:, 0] = 0
    return vn_ie


def mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(vn: np.array, vt: np.array) -> np.array:
    z_kin_hor_e = 0.5 * (vn * vn + vt * vt)
    z_kin_hor_e[:, 0] = 0
    return z_kin_hor_e


def mo_velocity_advection_stencil_02_numpy(
    grid, wgtfac_e: np.array, vn: np.array, vt: np.array, **kwargs
) -> tuple:
    vn_ie = mo_velocity_advection_stencil_02_vn_ie_numpy(wgtfac_e, vn)
    z_kin_hor_e = mo_velocity_advection_stencil_02_z_kin_hor_e_numpy(vn, vt)
    return (
        vn_ie,
        z_kin_hor_e,
    )


class TestMoVelocityAdvectionStencil02VnIe(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_02
    OUTPUTS = ("vn_ie", "z_kin_hor_e")

    @classmethod
    def reference(cls, grid, wgtfac_e: np.array, vn: np.array, vt: np.array, **kwargs) -> dict:
        vn_ie, z_kin_hor_e = mo_velocity_advection_stencil_02_numpy(grid, wgtfac_e, vn, vt)
        return dict(
            vn_ie=vn_ie[int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)],
            z_kin_hor_e=z_kin_hor_e[
                int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)
            ],
        )

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_e = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        vn = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        vt = random_field(grid, EdgeDim, KDim, dtype=vpfloat)

        vn_ie = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, EdgeDim, KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie[int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)],
            z_kin_hor_e=z_kin_hor_e[
                int32(1) : int32(grid.num_cells), int32(1) : int32(grid.num_levels)
            ],
            horizontal_start=int32(1),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
