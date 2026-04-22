# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    compute_horizontal_kinetic_energy,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def compute_horizontal_kinetic_energy_numpy(vn: np.ndarray, vt: np.ndarray) -> tuple:
    vn_ie = vn
    z_vt_ie = vt
    z_kin_hor_e = 0.5 * ((vn * vn) + (vt * vt))
    return vn_ie, z_vt_ie, z_kin_hor_e


class TestComputeHorizontalKineticEnergy(stencil_tests.StencilTest):
    PROGRAM = compute_horizontal_kinetic_energy
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        vn: np.ndarray,
        vt: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        vn_ie, z_vt_ie, z_kin_hor_e = compute_horizontal_kinetic_energy_numpy(vn, vt)
        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vt = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)

        vn_ie = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_vt_ie = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_kin_hor_e = self.data_alloc.zero_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)

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
