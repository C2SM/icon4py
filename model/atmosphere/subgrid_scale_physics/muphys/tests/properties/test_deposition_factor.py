# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import deposition_factor
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn, idx

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestIceDepositionNucleation(StencilTest):
    PROGRAM = deposition_factor
    OUTPUTS = ("deposition_factor",)

    @staticmethod
    def reference(grid, t: np.array, qvsi: np.array, QMIN: wpfloat, ALS: wpfloat, RD: wpfloat, RV: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(deposition_factor=np.full(t.shape, 1.3234329478493952e-05))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t               = constant_field(grid, 272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            qvsi            = constant_field(grid, 0.00416891, dims.CellDim, dims.KDim, dtype=wpfloat),
            QMIN            = graupel_ct.qmin,
            ALS             = thermodyn.als,
            RD              = thermodyn.rd,
            RV              = thermodyn.rv,
            TMELT           = thermodyn.tmelt,
            deposition_factor = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
