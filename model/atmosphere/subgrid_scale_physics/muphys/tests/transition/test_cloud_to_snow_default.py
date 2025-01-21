# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions.cloud_to_snow import cloud_to_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestCloudToSnowDefault(StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qs: np.array, ns: np.array, lam: np.array, V1S: wpfloat, V0S: wpfloat, TFRZ_HOM: wpfloat, QMIN: wpfloat, **kwargs) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 281.787, dims.CellDim, dtype=wpfloat),
            qc               = constant_field(grid, 0.0, dims.CellDim, dtype=wpfloat),
            qs               = constant_field(grid, 3.63983e-40, dims.CellDim, dtype=wpfloat),
            ns               = constant_field(grid, 800000.0, dims.CellDim, dtype=wpfloat),
            lam              = constant_field(grid, 1.0e+10, dims.CellDim, dtype=wpfloat),
            V1S              = graupel_ct.v1s,
            V0S              = graupel_ct.v0s,
            TFRZ_HOM         = graupel_ct.tfrz_hom,
            QMIN             = graupel_ct.qmin,
            riming_snow_rate = constant_field(grid, 0., dims.CellDim, dtype=wpfloat)
        )

