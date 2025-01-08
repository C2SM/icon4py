# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.saturation_adjustment import saturation_adjustment
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestSaturationAdjustment(StencilTest):
    PROGRAM = saturation_adjustment
    OUTPUTS = ("te","qve","qce")

    @staticmethod
    def reference(grid, te: np.array, qve: np.array, qce: np.array, qre: np.array, qti: np.array, rho: np.array, CI: wpfloat, CLW: wpfloat, CVD: wpfloat, CVV: wpfloat, LVC: wpfloat, TMELT: wpfloat, RV: wpfloat, **kwargs) -> dict:
        return dict(qce=np.full(te.shape, 9.5724552280369163E-007))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            te          = constant_field(grid, 273.90911754406039, dims.CellDim, dtype=wpfloat),
            qve         = constant_field(grid, 4.4913424511676030E-003, dims.CellDim, dtype=wpfloat),
            qce         = constant_field(grid, 6.0066941654987605E-013, dims.CellDim, dtype=wpfloat),
            qre         = constant_field(grid, 2.5939378002267028E-004, dims.CellDim, dtype=wpfloat),
            qti         = constant_field(grid, 1.0746937601645517E-005, dims.CellDim, dtype=wpfloat),
            rho         = constant_field(grid, 1.1371657035251757, dims.CellDim, dtype=wpfloat),
            CI          = graupel_ct.ci,
            CLW         = thermodyn.clw,
            CVD         = thermodyn.cvd,
            CVV         = thermodyn.cvv,
            LVC         = graupel_ct.lvc,
            TMELT       = thermodyn.tmelt,
            RV          = thermodyn.rv,
            te_out      = constant_field(grid, 0., dims.CellDim, dtype=wpfloat),
            qve_out     = constant_field(grid, 0., dims.CellDim, dtype=wpfloat),
            qce_out     = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
