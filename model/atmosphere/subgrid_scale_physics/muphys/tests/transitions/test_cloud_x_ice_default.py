# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_x_ice
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestCloudXIceDefault(StencilTest):
    PROGRAM = cloud_x_ice
    OUTPUTS = ("freezing_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qi: np.array, dt: wpfloat, TFRZ_HOM: wpfloat, QMIN: wpfloat, TMELT: wpfloat, **kwargs) -> dict:
        return dict(freezing_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t             = constant_field(grid, 256.835, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc            = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi            = constant_field(grid, 4.50245e-7, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt            = 30.0,
            TFRZ_HOM      = graupel_ct.tfrz_hom,
            QMIN          = graupel_ct.qmin,
            TMELT         = thermodyn.tmelt,
            freezing_rate = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )
