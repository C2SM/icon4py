# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.common import dimension as dims
import numpy as xp


class TestComputePpm4gpuParabolaCoefficients(helpers.StencilTest):
    PROGRAM = compute_ppm4gpu_parabola_coefficients
    OUTPUTS = ("z_delta_q", "z_a1")

    @staticmethod
    def reference(
        grid, z_face_up: xp.ndarray, z_face_low: xp.ndarray, p_cc: xp.ndarray, **kwargs
    ) -> dict:
        z_delta_q = 0.5 * (z_face_up - z_face_low)
        z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)
        return dict(z_delta_q=z_delta_q, z_a1=z_a1)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        z_face_up = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_face_low = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_delta_q = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_a1 = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            z_face_up=z_face_up,
            z_face_low=z_face_low,
            p_cc=p_cc,
            z_delta_q=z_delta_q,
            z_a1=z_a1,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
