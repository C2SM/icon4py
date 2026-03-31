# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_rain
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestCloudToRain(StencilTest):
    PROGRAM = cloud_to_rain
    OUTPUTS = ("conversion_rate",)

    @static_reference
    def reference(
        grid: base.Grid, t: np.ndarray, qc: np.ndarray, qr: np.ndarray, nc: np.ndarray, **kwargs
    ) -> dict:
        return dict(conversion_rate=np.full(t.shape, 0.0045484481075162512))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(267.25, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.constant_field(5.52921e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=self.data_alloc.constant_field(2.01511e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            nc=100.0,
            conversion_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
