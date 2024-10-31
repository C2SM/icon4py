# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.diffusion.stencils.temporary_field_for_grid_point_cold_pools_enhancement import (
    temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestTemporaryFieldForGridPointColdPoolsEnhancement(StencilTest):
    PROGRAM = temporary_field_for_grid_point_cold_pools_enhancement
    OUTPUTS = ("enh_diffu_3d",)

    @staticmethod
    def reference(
        grid, theta_v: xp.array, theta_ref_mc: xp.array, thresh_tdiff, smallest_vpfloat, **kwargs
    ) -> dict:
        theta_v = xp.asarray(theta_v)
        theta_ref_mc = xp.asarray(theta_ref_mc)
        thresh_tdiff = xp.asarray(thresh_tdiff)
        smallest_vpfloat = xp.asarray(smallest_vpfloat)
        c2e2c = xp.asarray(grid.connectivities[dims.C2E2CDim])
        tdiff = (
            theta_v
            - xp.sum(xp.where((c2e2c != -1)[:, :, xp.newaxis], theta_v[c2e2c], 0), axis=1) / 3
        )
        trefdiff = (
            theta_ref_mc
            - xp.sum(xp.where((c2e2c != -1)[:, :, xp.newaxis], theta_ref_mc[c2e2c], 0), axis=1) / 3
        )

        enh_diffu_3d = xp.where(
            ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0)
            | (tdiff - trefdiff < 1.5 * thresh_tdiff),
            (thresh_tdiff - tdiff + trefdiff) * 5e-4,
            smallest_vpfloat,
        )

        return dict(enh_diffu_3d=enh_diffu_3d)

    @pytest.fixture
    def input_data(self, grid):
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_ref_mc = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        enh_diffu_3d = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        thresh_tdiff = wpfloat("5.0")
        smallest_vpfloat = -xp.finfo(vpfloat).max

        return dict(
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            enh_diffu_3d=enh_diffu_3d,
            thresh_tdiff=thresh_tdiff,
            smallest_vpfloat=smallest_vpfloat,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
