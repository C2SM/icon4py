# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    compute_horizontal_kinetic_energy,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


def compute_horizontal_kinetic_energy_numpy(vn: np.array, vt: np.array) -> tuple:
    vn_ie = vn
    z_vt_ie = vt
    z_kin_hor_e = 0.5 * ((vn * vn) + (vt * vt))
    return vn_ie, z_vt_ie, z_kin_hor_e


class TestComputeHorizontalKineticEnergy(StencilTest):
    PROGRAM = compute_horizontal_kinetic_energy
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(grid, vn: np.array, vt: np.array, **kwargs) -> dict:
        vn_ie, z_vt_ie, z_kin_hor_e = compute_horizontal_kinetic_energy_numpy(vn, vt)
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        vn_ie = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_vt_ie = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_kin_hor_e = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
