# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_03a import (
    upwind_hflux_miura_cycl_stencil_03a,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestUpwindHfluxMiuraCyclStencil03a(StencilTest):
    PROGRAM = upwind_hflux_miura_cycl_stencil_03a
    OUTPUTS = ("p_out_e",)

    @staticmethod
    def reference(
        grid,
        z_tracer_mflx_1_dsl: np.array,
        z_tracer_mflx_2_dsl: np.array,
        **kwargs,
    ):
        p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl) / float(2)
        return dict(p_out_e=p_out_e)

    @pytest.fixture
    def input_data(self, grid):
        z_tracer_mflx_1_dsl = random_field(grid, dims.EdgeDim, dims.KDim)
        z_tracer_mflx_2_dsl = random_field(grid, dims.EdgeDim, dims.KDim)
        p_out_e = zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            z_tracer_mflx_1_dsl=z_tracer_mflx_1_dsl,
            z_tracer_mflx_2_dsl=z_tracer_mflx_2_dsl,
            p_out_e=p_out_e,
        )
