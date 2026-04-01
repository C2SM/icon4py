# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    calculate_nabla2_of_theta,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


def calculate_nabla2_of_theta_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    z_nabla2_e: np.ndarray,
    geofac_div: np.ndarray,
) -> np.ndarray:
    c2e = connectivities[dims.C2E]
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_temp = np.sum(z_nabla2_e[c2e] * geofac_div, axis=1)  # sum along edge dimension
    return z_temp


class TestCalculateNabla2OfTheta(stencil_tests.StencilTest):
    PROGRAM = calculate_nabla2_of_theta
    OUTPUTS = ("z_temp",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        z_nabla2_e: np.ndarray,
        geofac_div: np.ndarray,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        z_temp = calculate_nabla2_of_theta_numpy(connectivities, z_nabla2_e, geofac_div)
        return dict(z_temp=z_temp)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        z_nabla2_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        geofac_div = self.data_alloc.random_field(dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)

        z_temp = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            z_nabla2_e=z_nabla2_e,
            geofac_div=geofac_div,
            z_temp=z_temp,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
