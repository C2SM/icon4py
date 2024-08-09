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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.v_limit_prbl_sm_stencil_02 import v_limit_prbl_sm_stencil_02
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)


class TestVLimitPrblSmStencil02(StencilTest):
    PROGRAM = v_limit_prbl_sm_stencil_02
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
            l_limit=l_limit, p_cc=p_cc, p_face=p_face, p_face_up=p_face_up, p_face_low=p_face_low
        )
