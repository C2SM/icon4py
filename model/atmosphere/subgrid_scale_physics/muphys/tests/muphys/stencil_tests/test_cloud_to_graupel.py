# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestCloudToGraupel(StencilTest):
    PROGRAM = cloud_to_graupel
    OUTPUTS = ("riming_graupel_rate",)

    @static_reference
    def reference(
        grid: base.Grid, t: np.ndarray, rho: np.ndarray, qc: np.ndarray, qg: np.ndarray, **kwargs
    ) -> dict:
        return dict(riming_graupel_rate=np.full(t.shape, 2.7054723496793982e-10))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(256.983, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(0.909677, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.constant_field(8.60101e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=self.data_alloc.constant_field(4.11575e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_graupel_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
