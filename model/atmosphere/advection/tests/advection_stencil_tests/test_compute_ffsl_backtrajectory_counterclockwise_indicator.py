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

import icon4py.model.testing.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory_counterclockwise_indicator import (
    compute_ffsl_backtrajectory_counterclockwise_indicator,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc


class TestComputeFfslBacktrajectoryCounterclockwiseIndicator(helpers.StencilTest):
    PROGRAM = compute_ffsl_backtrajectory_counterclockwise_indicator
    OUTPUTS = ("lvn_sys_pos",)

    @staticmethod
    def reference(
        grid, p_vn: np.array, tangent_orientation: np.array, lcounterclock: bool, **kwargs
    ) -> dict:
        tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

        tangent_orientation = np.broadcast_to(tangent_orientation, p_vn.shape)

        lvn_sys_pos_true = np.where(tangent_orientation * p_vn >= 0.0, True, False)

        mask_lcounterclock = np.broadcast_to(lcounterclock, p_vn.shape)

        lvn_sys_pos = np.where(mask_lcounterclock, lvn_sys_pos_true, False)

        return dict(lvn_sys_pos=lvn_sys_pos)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        lvn_sys_pos = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=bool)
        lcounterclock = True
        return dict(
            p_vn=p_vn,
            tangent_orientation=tangent_orientation,
            lvn_sys_pos=lvn_sys_pos,
            lcounterclock=lcounterclock,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
