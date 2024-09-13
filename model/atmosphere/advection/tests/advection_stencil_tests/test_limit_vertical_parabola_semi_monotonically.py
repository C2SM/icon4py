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

from icon4py.model.atmosphere.advection.stencils.limit_vertical_parabola_semi_monotonically import (
    limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)


class TestLimitVerticalParabolaSemiMonotonically(StencilTest):
    PROGRAM = limit_vertical_parabola_semi_monotonically
    OUTPUTS = ("p_face_up", "p_face_low")

    @staticmethod
    def reference(grid, l_limit: np.array, p_face: np.array, p_cc: np.array, **kwargs):
        q_face_up, q_face_low = np.where(
            l_limit != 0,
            np.where(
                (p_cc < np.minimum(p_face[:, :-1], p_face[:, 1:])),
                (p_cc, p_cc),
                np.where(
                    p_face[:, :-1] > p_face[:, 1:],
                    (3.0 * p_cc - 2.0 * p_face[:, 1:], p_face[:, 1:]),
                    (p_face[:, :-1], 3.0 * p_cc - 2.0 * p_face[:, :-1]),
                ),
            ),
            (p_face[:, :-1], p_face[:, 1:]),
        )
        return dict(p_face_up=q_face_up, p_face_low=q_face_low)

    @pytest.fixture
    def input_data(self, grid):
        l_limit = random_mask(grid, dims.CellDim, dims.KDim, dtype=int32)
        p_cc = random_field(grid, dims.CellDim, dims.KDim)
        p_face = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        p_face_up = zero_field(grid, dims.CellDim, dims.KDim)
        p_face_low = zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            l_limit=l_limit,
            p_cc=p_cc,
            p_face=p_face,
            p_face_up=p_face_up,
            p_face_low=p_face_low,
            horizontal_start=0,
            horizontal_end=int32(grid.num_cells),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
