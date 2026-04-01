# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.temporary_field_for_grid_point_cold_pools_enhancement import (
    temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


@pytest.mark.embedded_remap_error
class TestTemporaryFieldForGridPointColdPoolsEnhancement(StencilTest):
    PROGRAM = temporary_field_for_grid_point_cold_pools_enhancement
    OUTPUTS = ("enh_diffu_3d",)

    @static_reference
    def reference(
        grid: base.Grid,
        theta_v: np.ndarray,
        theta_ref_mc: np.ndarray,
        thresh_tdiff,
        smallest_vpfloat,
        **kwargs,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        c2e2c = connectivities[dims.C2E2CDim]
        tdiff = (
            theta_v
            - np.sum(np.where((c2e2c != -1)[:, :, np.newaxis], theta_v[c2e2c], 0), axis=1) / 3
        )
        trefdiff = (
            theta_ref_mc
            - np.sum(np.where((c2e2c != -1)[:, :, np.newaxis], theta_ref_mc[c2e2c], 0), axis=1) / 3
        )

        enh_diffu_3d = np.where(
            ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0)
            | (tdiff - trefdiff < 1.5 * thresh_tdiff),
            (thresh_tdiff - tdiff + trefdiff) * 5e-4,
            smallest_vpfloat,
        )

        return dict(enh_diffu_3d=enh_diffu_3d)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        theta_v = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_ref_mc = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        enh_diffu_3d = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        thresh_tdiff = wpfloat("5.0")
        smallest_vpfloat = -np.finfo(vpfloat).max

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
