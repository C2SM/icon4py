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

from icon4py.model.atmosphere.dycore.compute_contravariant_correction import (
    compute_contravariant_correction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_contravariant_correction_numpy(
    vn: np.array, ddxn_z_full: np.array, ddxt_z_full: np.array, vt: np.array
) -> np.array:
    z_w_concorr_me = vn * ddxn_z_full + vt * ddxt_z_full
    return z_w_concorr_me


class TestComputeContravariantCorrection(StencilTest):
    PROGRAM = compute_contravariant_correction
    OUTPUTS = ("z_w_concorr_me",)

    @staticmethod
    def reference(
        grid,
        vn: np.array,
        ddxn_z_full: np.array,
        ddxt_z_full: np.array,
        vt: np.array,
        **kwargs,
    ) -> dict:
        z_w_concorr_me = compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, vt)
        return dict(z_w_concorr_me=z_w_concorr_me)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ddxn_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        ddxt_z_full = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_w_concorr_me = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            vn=vn,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            vt=vt,
            z_w_concorr_me=z_w_concorr_me,
            horizontal_start=0,
            horizontal_end=int32(grid.num_edges),
            vertical_start=0,
            vertical_end=int32(grid.num_levels),
        )
