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
from gt4py.next import as_field

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ppm_slope import (
    compute_ppm_slope,
)
from icon4py.model.common import dimension as dims


class TestComputePpmSlope(helpers.StencilTest):
    PROGRAM = compute_ppm_slope
    OUTPUTS = (
        helpers.Output(
            "z_slope", refslice=(slice(None), slice(None, -1)), gtslice=(slice(None), slice(1, -1))
        ),
    )

    @staticmethod
    def reference(
        grid, p_cc: np.array, p_cellhgt_mc_now: np.array, k: np.array, elev: gtx.int32, **kwargs
    ):
        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 2:] - p_cc[:, 1:-1]) / (p_cellhgt_mc_now[:, 2:] + p_cellhgt_mc_now[:, 1:-1])
        z_slope_a = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 2:])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 2:]) * zfac_m1
        )

        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 1:-1] - p_cc[:, 1:-1]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1]
        )
        z_slope_b = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 1:-1]) * zfac_m1
        )

        z_slope = np.where(k[1:-1] < elev, z_slope_a, z_slope_b)
        return dict(z_slope=z_slope)

    @pytest.fixture
    def input_data(self, grid):
        z_slope = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        p_cellhgt_mc_now = helpers.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        k = as_field(
            (dims.KDim,),
            np.arange(
                0, helpers._shape(grid, dims.KDim, extend={dims.KDim: 1})[0], dtype=gtx.int32
            ),
        )
        elev = k[-2].as_scalar()
        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            k=k,
            elev=elev,
            z_slope=z_slope,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels - 1),
        )
