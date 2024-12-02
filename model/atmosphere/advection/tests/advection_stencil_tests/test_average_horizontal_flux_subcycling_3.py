# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.average_horizontal_flux_subcycling_3 import (
    average_horizontal_flux_subcycling_3,
)
from icon4py.model.common import dimension as dims


class TestAverageHorizontalFluxSubcycling3(helpers.StencilTest):
    PROGRAM = average_horizontal_flux_subcycling_3
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        z_tracer_mflx_1_dsl: np.array,
        z_tracer_mflx_2_dsl: np.array,
        z_tracer_mflx_3_dsl: np.array,
        **kwargs,
    ) -> dict:
        p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl + z_tracer_mflx_3_dsl) / float(3)
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        z_tracer_mflx_1_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_2_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_3_dsl = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = helpers.zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_tracer_mflx_1_dsl=z_tracer_mflx_1_dsl,
            z_tracer_mflx_2_dsl=z_tracer_mflx_2_dsl,
            z_tracer_mflx_3_dsl=z_tracer_mflx_3_dsl,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
