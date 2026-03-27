# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestIceToGraupel(StencilTest):
    PROGRAM = ice_to_graupel
    OUTPUTS = ("aggregation",)

    @static_reference
    def reference(
        grid,
        rho: np.ndarray,
        qr: np.ndarray,
        qg: np.ndarray,
        qi: np.ndarray,
        sticking_eff: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(aggregation=np.full(rho.shape, 7.1049436957697864e-19))

    @input_data_fixture
    def input_data(self, grid):
        return dict(
            rho=self.data_alloc.constant_field(1.04848, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=self.data_alloc.constant_field(6.00408e-13, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=self.data_alloc.constant_field(1.19022e-18, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=self.data_alloc.constant_field(1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff=self.data_alloc.constant_field(
                1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            aggregation=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
