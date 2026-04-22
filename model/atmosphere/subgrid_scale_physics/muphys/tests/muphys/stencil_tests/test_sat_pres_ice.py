# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import sat_pres_ice
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestSatPresIce(stencil_tests.StencilTest):
    PROGRAM = sat_pres_ice
    OUTPUTS = ("pressure",)

    @stencil_tests.static_reference
    def reference(grid: base.Grid, t: np.ndarray, **kwargs) -> dict:
        return dict(pressure=np.full(t.shape, 1216.7746246067475))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            pressure=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
