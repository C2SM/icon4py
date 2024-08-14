# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.advection.hflux_ffsl_hybrid_stencil_02 import (
    hflux_ffsl_hybrid_stencil_02,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestHfluxFfslHybridStencil02(StencilTest):
    PROGRAM = hflux_ffsl_hybrid_stencil_02
    OUTPUTS = ("p_out_e_hybrid_2",)

    @staticmethod
    def reference(
        grid,
        p_out_e_hybrid_2: np.array,
        p_mass_flx_e: np.array,
        z_dreg_area: np.array,
        **kwargs,
    ):
        p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

        return dict(p_out_e_hybrid_2=p_out_e_hybrid_2)

    @pytest.fixture
    def input_data(self, grid):
        p_out_e_hybrid_2 = random_field(grid, EdgeDim, KDim)
        p_mass_flx_e = random_field(grid, EdgeDim, KDim)
        z_dreg_area = random_field(grid, EdgeDim, KDim)
        return dict(
            p_mass_flx_e=p_mass_flx_e,
            z_dreg_area=z_dreg_area,
            p_out_e_hybrid_2=p_out_e_hybrid_2,
        )
