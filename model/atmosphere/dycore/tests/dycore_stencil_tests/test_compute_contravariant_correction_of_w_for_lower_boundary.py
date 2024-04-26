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

from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w_for_lower_boundary import (
    compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def compute_contravariant_correction_of_w_for_lower_boundary_numpy(
    grid, e_bln_c_s: np.array, z_w_concorr_me: np.array, wgtfacq_c: np.array
) -> np.array:
    c2e = grid.connectivities[C2EDim]
    c2e_shape = c2e.shape
    c2ce_table = np.arange(c2e_shape[0] * c2e_shape[1]).reshape(c2e_shape)

    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_me_offset_1 = np.roll(z_w_concorr_me, shift=1, axis=1)
    z_w_concorr_me_offset_2 = np.roll(z_w_concorr_me, shift=2, axis=1)
    z_w_concorr_me_offset_3 = np.roll(z_w_concorr_me, shift=3, axis=1)

    z_w_concorr_mc_m1 = np.sum(e_bln_c_s[c2ce_table] * z_w_concorr_me_offset_1[c2e], axis=1)
    z_w_concorr_mc_m2 = np.sum(e_bln_c_s[c2ce_table] * z_w_concorr_me_offset_2[c2e], axis=1)
    z_w_concorr_mc_m3 = np.sum(e_bln_c_s[c2ce_table] * z_w_concorr_me_offset_3[c2e], axis=1)

    w_concorr_c = np.zeros_like(wgtfacq_c)
    w_concorr_c[:, -1] = (
        np.roll(wgtfacq_c, shift=1, axis=1) * z_w_concorr_mc_m1
        + np.roll(wgtfacq_c, shift=2, axis=1) * z_w_concorr_mc_m2
        + np.roll(wgtfacq_c, shift=3, axis=1) * z_w_concorr_mc_m3
    )[:, -1]

    return w_concorr_c


class TestComputeContravariantCorrectionOfWForLowerBoundary(StencilTest):
    PROGRAM = compute_contravariant_correction_of_w_for_lower_boundary
    OUTPUTS = ("w_concorr_c",)

    @staticmethod
    def reference(
        grid,
        e_bln_c_s: np.array,
        z_w_concorr_me: np.array,
        wgtfacq_c: np.array,
        **kwargs,
    ) -> dict:
        w_concorr_c = compute_contravariant_correction_of_w_for_lower_boundary_numpy(
            grid, e_bln_c_s, z_w_concorr_me, wgtfacq_c
        )
        return dict(w_concorr_c=w_concorr_c)

    @pytest.fixture
    def input_data(self, grid):
        e_bln_c_s = random_field(grid, CEDim, dtype=wpfloat)
        z_w_concorr_me = random_field(grid, EdgeDim, KDim, dtype=vpfloat)
        wgtfacq_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        w_concorr_c = zero_field(grid, CellDim, KDim, dtype=vpfloat)

        return dict(
            e_bln_c_s=e_bln_c_s,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_c=wgtfacq_c,
            w_concorr_c=w_concorr_c,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(grid.num_levels - 1),
            vertical_end=int32(grid.num_levels),
        )
