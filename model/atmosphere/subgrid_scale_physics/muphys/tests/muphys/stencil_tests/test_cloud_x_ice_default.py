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
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestCloudXIceDefault(StencilTest):
    PROGRAM = cloud_x_ice
    OUTPUTS = ("freezing_rate",)

    @static_reference
    def reference(
        grid, t: np.ndarray, qc: np.ndarray, qi: np.ndarray, dt: wpfloat, **kwargs
    ) -> dict:
        return dict(freezing_rate=np.full(t.shape, 0.0))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            t=self.data_alloc.constant_field(256.835, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=self.data_alloc.constant_field(4.50245e-7, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            freezing_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
