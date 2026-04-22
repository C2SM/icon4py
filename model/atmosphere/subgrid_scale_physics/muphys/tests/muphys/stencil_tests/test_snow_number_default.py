# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import snow_number
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestSnowNumberDefault(stencil_tests.StencilTest):
    PROGRAM = snow_number
    OUTPUTS = ("number",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid, t: np.ndarray, rho: np.ndarray, qs: np.ndarray, **kwargs
    ) -> dict:
        return dict(number=np.full(t.shape, 8.00e5))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(276.302, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.17797, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=self.data_alloc.constant_field(8.28451e-24, dims.CellDim, dims.KDim, dtype=wpfloat),
            number=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
