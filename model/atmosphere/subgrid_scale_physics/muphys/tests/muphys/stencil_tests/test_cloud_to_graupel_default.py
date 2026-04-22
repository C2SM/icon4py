# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestCloudToGraupelDefault(stencil_tests.StencilTest):
    PROGRAM = cloud_to_graupel
    OUTPUTS = ("riming_graupel_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid, t: np.ndarray, rho: np.ndarray, qc: np.ndarray, qg: np.ndarray, **kwargs
    ) -> dict:
        return dict(riming_graupel_rate=np.full(t.shape, 0.0))

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=self.data_alloc.constant_field(1.24783, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=self.data_alloc.constant_field(1.03636e-25, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_graupel_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
