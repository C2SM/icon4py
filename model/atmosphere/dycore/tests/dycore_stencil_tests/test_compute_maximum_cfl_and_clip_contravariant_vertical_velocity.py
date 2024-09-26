# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from gt4py.next import gtx

from icon4py.model.atmosphere.dycore.compute_maximum_cfl_and_clip_contravariant_vertical_velocity import (
    compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
    mesh, ddqz_z_half: np.array, z_w_con_c: np.array, cfl_w_limit, dtime
) -> tuple:
    num_rows, num_cols = z_w_con_c.shape
    cfl_clipping = np.where(
        np.abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        np.ones([num_rows, num_cols]),
        np.zeros_like(z_w_con_c),
    )
    num_rows, num_cols = cfl_clipping.shape
    vcfl = np.where(cfl_clipping == 1.0, z_w_con_c * dtime / ddqz_z_half, 0.0)
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl < -0.85),
        -0.85 * ddqz_z_half / dtime,
        z_w_con_c,
    )
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c
    )

    return cfl_clipping, vcfl, z_w_con_c


class TestComputeMaximumCflAndClipContravariantVerticalVelocity(StencilTest):
    PROGRAM = compute_maximum_cfl_and_clip_contravariant_vertical_velocity
    OUTPUTS = ("cfl_clipping", "vcfl", "z_w_con_c")

    @staticmethod
    def reference(
        grid, ddqz_z_half: np.array, z_w_con_c: np.array, cfl_w_limit, dtime, **kwargs
    ) -> dict:
        (
            cfl_clipping,
            vcfl,
            z_w_con_c,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            grid, ddqz_z_half, z_w_con_c, cfl_w_limit, dtime
        )

        return dict(
            cfl_clipping=cfl_clipping,
            vcfl=vcfl,
            z_w_con_c=z_w_con_c,
        )

    @pytest.fixture
    def input_data(self, grid):
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_w_con_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        cfl_clipping = random_mask(grid, dims.CellDim, dims.KDim, dtype=bool)
        vcfl = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        cfl_w_limit = vpfloat("5.0")
        dtime = wpfloat("9.0")

        return dict(
            ddqz_z_half=ddqz_z_half,
            z_w_con_c=z_w_con_c,
            cfl_clipping=cfl_clipping,
            vcfl=vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
