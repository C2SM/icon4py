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

from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory_counterclockwise_indicator import (
    compute_ffsl_backtrajectory_counterclockwise_indicator,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestComputeFfslBacktrajectoryCounterclockwiseIndicator(StencilTest):
    PROGRAM = compute_ffsl_backtrajectory_counterclockwise_indicator
    OUTPUTS = ("lvn_sys_pos",)

    @staticmethod
    def reference(
        grid, lcounterclock: bool, p_vn: np.array, tangent_orientation: np.array, **kwargs
    ):
        tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

        tangent_orientation = np.broadcast_to(tangent_orientation, p_vn.shape)

        lvn_sys_pos_true = np.where(tangent_orientation * p_vn >= 0.0, True, False)

        mask_lcounterclock = np.broadcast_to(lcounterclock, p_vn.shape)

        lvn_sys_pos = np.where(mask_lcounterclock, lvn_sys_pos_true, False)

        return dict(lvn_sys_pos=lvn_sys_pos)

    @pytest.fixture
    def input_data(self, grid):
        lcounterclock = True
        p_vn = random_field(grid, dims.EdgeDim, dims.KDim)
        tangent_orientation = random_field(grid, dims.EdgeDim)
        lvn_sys_pos = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=bool)
        return dict(
            lcounterclock=lcounterclock,
            p_vn=p_vn,
            tangent_orientation=tangent_orientation,
            lvn_sys_pos=lvn_sys_pos,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
