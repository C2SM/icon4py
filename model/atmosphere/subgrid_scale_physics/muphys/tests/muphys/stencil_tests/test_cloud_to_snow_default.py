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
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestCloudToSnowDefault(StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @static_reference
    def reference(
        grid,
        t: np.ndarray,
        qc: np.ndarray,
        qs: np.ndarray,
        ns: np.ndarray,
        lam: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 0.0))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            t=self.data_alloc.constant_field(281.787, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=self.data_alloc.constant_field(3.63983e-40, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=self.data_alloc.constant_field(800000.0, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=self.data_alloc.constant_field(1.0e10, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
