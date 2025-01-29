# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestCloudToSnow(StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qs: np.array, ns: np.array, lam: np.array, V0S: wpfloat, V1S: wpfloat, TFRZ_HOM: wpfloat, QMIN: wpfloat, **kwargs) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 9.5431874564438999e-10))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t                = constant_field(grid, 256.571, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc               = constant_field(grid, 3.31476e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs               = constant_field(grid, 7.47365e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns               = constant_field(grid, 3.37707e+07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam              = constant_field(grid, 8989.78, dims.CellDim, dims.KDim, dtype=wpfloat),
            V0S              = graupel_ct.v0s,
            V1S              = graupel_ct.v1s,
            TFRZ_HOM         = graupel_ct.tfrz_hom,
            QMIN             = graupel_ct.qmin,
            riming_snow_rate = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat)
        )

