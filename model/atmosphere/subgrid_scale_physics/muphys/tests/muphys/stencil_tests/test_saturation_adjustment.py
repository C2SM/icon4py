# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import saturation_adjustment
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


@pytest.mark.embedded_only
class TestSaturationAdjustment(StencilTest):
    PROGRAM = saturation_adjustment
    OUTPUTS = ("te_out", "qve_out", "qce_out", "mask_out")

    @staticmethod
    def reference(
        grid,
        te: np.ndarray,
        qve: np.ndarray,
        qce: np.ndarray,
        qre: np.ndarray,
        qti: np.ndarray,
        rho: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(
            te_out=np.full(te.shape, 273.91226488486984),
            qve_out=np.full(te.shape, 4.4903852062454690e-003),
            qce_out=np.full(te.shape, 9.5724552280369163e-007),
            mask_out=np.full(te.shape, False),
        )

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            te=data_alloc.constant_field(
                grid, 273.90911754406039, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qve=data_alloc.constant_field(
                grid, 4.4913424511676030e-003, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qce=data_alloc.constant_field(
                grid, 6.0066941654987605e-013, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qre=data_alloc.constant_field(
                grid, 2.5939378002267028e-004, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qti=data_alloc.constant_field(
                grid, 1.0746937601645517e-005, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            rho=data_alloc.constant_field(
                grid, 1.1371657035251757, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            te_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qve_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qce_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            mask_out=data_alloc.constant_field(grid, True, dims.CellDim, dims.KDim, dtype=bool),
        )
