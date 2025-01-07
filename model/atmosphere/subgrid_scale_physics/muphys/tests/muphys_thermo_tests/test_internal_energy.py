# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.internal_energy import internal_energy
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestInternalEnergy(StencilTest):
    PROGRAM = internal_energy
    OUTPUTS = ("internal_energy",)

    @staticmethod
    def reference(grid, t: np.array, qv: np.array, qliq: np.array, qice: np.array, rho: np.array, dz: np.array, CI: wpfloat, CLW: wpfloat, CVD: wpfloat, CVV: wpfloat, LSC: wpfloat, LVC: wpfloat, **kwargs) -> dict:
        return dict(internal_energy=np.full(t.shape, 38265357.270336017))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 255.756, dims.CellDim, dtype=wpfloat),
            qv               = constant_field(grid, 0.00122576, dims.CellDim, dtype=wpfloat),
            qliq             = constant_field(grid, 1.63837e-20, dims.CellDim, dtype=wpfloat),
            qice             = constant_field(grid, 1.09462e-08, dims.CellDim, dtype=wpfloat),
            rho              = constant_field(grid, 0.83444, dims.CellDim, dtype=wpfloat),
            dz               = constant_field(grid, 249.569, dims.CellDim, dtype=wpfloat),
            CI               = graupel_ct.ci,
            CLW              = thermodyn.clw,
            CVD              = thermodyn.cvd,
            CVV              = thermodyn.cvv,
            LSC              = graupel_ct.lsc,
            LVC              = graupel_ct.lvc,
            internal_energy  = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
