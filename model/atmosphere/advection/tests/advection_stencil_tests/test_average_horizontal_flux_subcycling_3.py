# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.stencils.average_horizontal_flux_subcycling_3 import (
    average_horizontal_flux_subcycling_3,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestAverageHorizontalFluxSubcycling3(StencilTest):
    PROGRAM = average_horizontal_flux_subcycling_3
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        z_tracer_mflx_1_dsl: np.array,
        z_tracer_mflx_2_dsl: np.array,
        z_tracer_mflx_3_dsl: np.array,
        **kwargs,
    ):
        p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl + z_tracer_mflx_3_dsl) / float(3)
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid):
        z_tracer_mflx_1_dsl = random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_2_dsl = random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_3_dsl = random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_tracer_mflx_1_dsl=z_tracer_mflx_1_dsl,
            z_tracer_mflx_2_dsl=z_tracer_mflx_2_dsl,
            z_tracer_mflx_3_dsl=z_tracer_mflx_3_dsl,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )