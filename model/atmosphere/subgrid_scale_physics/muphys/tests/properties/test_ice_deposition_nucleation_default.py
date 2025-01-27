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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import ice_deposition_nucleation
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import graupel_ct, thermodyn, idx

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field
from icon4py.model.common.type_alias import wpfloat


class TestIceDepositionNucleationDefault(StencilTest):
    PROGRAM = ice_deposition_nucleation
    OUTPUTS = ("vapor_deposition_rate",)

    @staticmethod
    def reference(grid, t: np.array, qc: np.array, qi: np.array, ni: np.array, dvsi: np.array, dt: wpfloat, QMIN: wpfloat, M0_ICE: wpfloat, TFRZ_HET1: wpfloat, TFRZ_HET2: wpfloat, **kwargs) -> dict:
        return dict(vapor_deposition_rate=np.full(t.shape, 0.0))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            t               = constant_field(grid, 272.731, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc              = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi              = constant_field(grid, 2.02422e-23, dims.CellDim, dims.KDim, dtype=wpfloat),
            ni              = constant_field(grid, 5.05089, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsi            = constant_field(grid, -0.000618828, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt              = 30.0,
            QMIN            = graupel_ct.qmin,
            M0_ICE          = graupel_ct.m0_ice,
            TFRZ_HET1       = graupel_ct.tfrz_het1,
            TFRZ_HET2       = graupel_ct.tfrz_het2,
            vapor_deposition_rate = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
