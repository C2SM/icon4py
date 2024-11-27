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
from icon4py.model.atmosphere.advection.stencils.compute_ppm_quartic_face_values import (
    compute_ppm_quartic_face_values,
)
from icon4py.model.common import dimension as dims
import numpy as xp


class TestComputePpmQuarticFaceValues(helpers.StencilTest):
    PROGRAM = compute_ppm_quartic_face_values
    OUTPUTS = (helpers.Output("p_face", gtslice=(slice(None), slice(2, None))),)

    @staticmethod
    def reference(
        grid, p_cc: xp.array, p_cellhgt_mc_now: xp.array, z_slope: xp.array, **kwargs
    ) -> dict:
        p_cellhgt_mc_now_k_minus_1 = p_cellhgt_mc_now[:, 1:-2]
        p_cellhgt_mc_now_k_minus_2 = p_cellhgt_mc_now[:, 0:-3]
        p_cellhgt_mc_now_k_plus_1 = p_cellhgt_mc_now[:, 3:]
        p_cellhgt_mc_now = p_cellhgt_mc_now[:, 2:-1]

        p_cc_k_minus_1 = p_cc[:, 1:-1]
        p_cc = p_cc[:, 2:]
        z_slope_k_minus_1 = z_slope[:, 1:-1]
        z_slope = z_slope[:, 2:]

        zgeo1 = p_cellhgt_mc_now_k_minus_1 / (p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now)
        zgeo2 = 1.0 / (
            p_cellhgt_mc_now_k_minus_2
            + p_cellhgt_mc_now_k_minus_1
            + p_cellhgt_mc_now
            + p_cellhgt_mc_now_k_plus_1
        )
        zgeo3 = (p_cellhgt_mc_now_k_minus_2 + p_cellhgt_mc_now_k_minus_1) / (
            2.0 * p_cellhgt_mc_now_k_minus_1 + p_cellhgt_mc_now
        )
        zgeo4 = (p_cellhgt_mc_now_k_plus_1 + p_cellhgt_mc_now) / (
            2 * p_cellhgt_mc_now + p_cellhgt_mc_now_k_minus_1
        )

        p_face = (
            p_cc_k_minus_1
            + zgeo1 * (p_cc - p_cc_k_minus_1)
            + zgeo2
            * (
                (2 * p_cellhgt_mc_now * zgeo1) * (zgeo3 - zgeo4) * (p_cc - p_cc_k_minus_1)
                - zgeo3 * p_cellhgt_mc_now_k_minus_1 * z_slope
                + zgeo4 * p_cellhgt_mc_now * z_slope_k_minus_1
            )
        )
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_cellhgt_mc_now = helpers.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        z_slope = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_face = helpers.zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            z_slope=z_slope,
            p_face=p_face,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=2,
            vertical_end=gtx.int32(grid.num_levels),
        )
