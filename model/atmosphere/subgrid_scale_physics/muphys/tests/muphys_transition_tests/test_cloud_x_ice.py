# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.cloud_x_ice import cloud_x_ice
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestCloudToSnow(StencilTest):
    PROGRAM = cloud_x_ice
    OUTPUTS = ("freezing_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qi: np.array, dt: wpfloat, tfrz_hom: wpfloat, qmin: wpfloat, t_melt: wpfloat, **kwargs) -> dict:
        return dict(freezing_rate=np.full(t.shape, -1.5008166666666666e-08))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t       = constant_field(grid, 256.835, dims.CellDim, dtype=wpfloat),
            qc      = constant_field(grid, 0.0, dims.CellDim, dtype=wpfloat),
            qi      = constant_field(grid, 4.50245e-07, dims.CellDim, dtype=wpfloat),
            dt      = constant_field(grid, 30.0, dims.CellDim, dtype=wpfloat),
            tfrz_hom= graupel_ct.tfrz_hom,
            qmin    = graupel_ct.qmin,
            t_melt  = thermodyn.tmelt,
            freezing_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )
