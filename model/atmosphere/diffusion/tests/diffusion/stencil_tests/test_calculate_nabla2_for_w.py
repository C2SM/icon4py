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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    calculate_nabla2_for_w,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def calculate_nabla2_for_w_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray], w: np.ndarray, geofac_n2s: np.ndarray
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CO]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    z_nabla2_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], w[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return z_nabla2_c


@pytest.mark.embedded_remap_error
class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = calculate_nabla2_for_w
    OUTPUTS = ("z_nabla2_c",)

    @static_reference
    def reference(
        grid: base.Grid,
        w: np.ndarray,
        geofac_n2s: np.ndarray,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        z_nabla2_c = calculate_nabla2_for_w_numpy(connectivities, w, geofac_n2s)
        return dict(z_nabla2_c=z_nabla2_c)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        w = self.data_alloc.constant_field(1.0, dims.CellDim, dims.KDim)
        geofac_n2s = self.data_alloc.constant_field(2.0, dims.CellDim, dims.C2E2CODim)
        z_nabla2_c = self.data_alloc.zero_field(dims.CellDim, dims.KDim)

        return dict(
            w=w,
            geofac_n2s=geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
