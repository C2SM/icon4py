# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    saturation_adjustment,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


if TYPE_CHECKING:
    from icon4py.model.common.grid import base, base as base_grid


class TestSaturationAdjustment(stencil_tests.StencilTest):
    PROGRAM = saturation_adjustment
    OUTPUTS = ("te_out", "qve_out", "qce_out")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        te: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(
            te_out=np.full(te.shape, 273.91226488486984),
            qve_out=np.full(te.shape, 4.4903852062454690e-003),
            qce_out=np.full(te.shape, 9.5724552280369163e-007),
        )

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        return dict(
            te=self.data_alloc.constant_field(
                273.90911754406039, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            q_in=Q(
                v=self.data_alloc.constant_field(
                    4.4913424511676030e-003, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                c=self.data_alloc.constant_field(
                    6.0066941654987605e-013, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                r=self.data_alloc.constant_field(
                    2.5939378002267028e-004, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                s=self.data_alloc.constant_field(
                    3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                i=self.data_alloc.constant_field(
                    3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                g=self.data_alloc.constant_field(
                    3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
            ),
            rho=self.data_alloc.constant_field(
                1.1371657035251757, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            te_out=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qve_out=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            qce_out=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
