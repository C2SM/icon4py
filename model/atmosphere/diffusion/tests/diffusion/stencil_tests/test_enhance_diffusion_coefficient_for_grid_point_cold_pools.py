# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestEnhanceDiffusionCoefficientForGridPointColdPools(StencilTest):
    PROGRAM = enhance_diffusion_coefficient_for_grid_point_cold_pools
    OUTPUTS = ("kh_smag_e",)

    @static_reference
    def reference(
        grid: base.Grid,
        kh_smag_e: np.ndarray,
        enh_diffu_3d: np.ndarray,
        **kwargs,
    ) -> dict:
        connectivities = grid.ndarray_connectivities
        e2c = connectivities[dims.E2CDim]
        kh_smag_e = np.maximum(
            kh_smag_e,
            np.max(
                np.where((e2c != -1)[:, :, np.newaxis], enh_diffu_3d[e2c], -math.inf),
                axis=1,
            ),
        )
        return dict(kh_smag_e=kh_smag_e)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        kh_smag_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        enh_diffu_3d = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            kh_smag_e=kh_smag_e,
            enh_diffu_3d=enh_diffu_3d,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
