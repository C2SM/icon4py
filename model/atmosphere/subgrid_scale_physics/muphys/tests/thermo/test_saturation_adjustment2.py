# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment2 import saturation_adjustment2

from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field
from icon4py.model.common.type_alias import wpfloat


class TestSaturationAdjustment(StencilTest):
    PROGRAM = saturation_adjustment2
    OUTPUTS = ("te_out", "qve_out", "qce_out")

    @staticmethod
    def reference(grid, te: np.array, qve: np.array, qce: np.array, qre: np.array, qti: np.array, cvc: np.array, ue: np.array, Tx_hold: np.array, Tx: np.array, rho: np.array, **kwargs) -> dict:
        return dict(te_out=np.full(te.shape, 273.91226488486984), qve_out=np.full(te.shape, 4.4903852062454690E-003), qce_out=np.full(te.shape, 9.5724552280369163E-007))

    @pytest.fixture
    def input_data(self, grid):

        return dict(
            te          = constant_field(grid, 273.90911754406039, dims.CellDim, dims.KDim, dtype=wpfloat),
            qve         = constant_field(grid, 4.4913424511676030E-003, dims.CellDim, dims.KDim, dtype=wpfloat),
            qce         = constant_field(grid, 6.0066941654987605E-013, dims.CellDim, dims.KDim, dtype=wpfloat),
            qre         = constant_field(grid, 2.5939378002267028E-004, dims.CellDim, dims.KDim, dtype=wpfloat),
            qti         = constant_field(grid, 1.0746937601645517E-005, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho         = constant_field(grid, 1.1371657035251757, dims.CellDim, dims.KDim, dtype=wpfloat),
            cvc         = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            ue          = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            Tx_hold     = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            Tx          = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            qx_hold     = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            qx          = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            dqx         = constant_field(grid, 0.0, dims.CellDim, dims.KDim, dtype=wpfloat),                     # Temporary
            te_out      = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat),
            qve_out     = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat),
            qce_out     = constant_field(grid, 0., dims.CellDim, dims.KDim, dtype=wpfloat),
        )
