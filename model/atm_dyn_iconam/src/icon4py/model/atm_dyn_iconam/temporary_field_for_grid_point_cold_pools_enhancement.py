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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum, where

from icon4py.model.common.dimension import C2E2C, C2E2CDim, CellDim, KDim


@field_operator
def _temporary_field_for_grid_point_cold_pools_enhancement(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
) -> Field[[CellDim, KDim], float]:
    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=C2E2CDim) / 3.0
    trefdiff = theta_ref_mc - neighbor_sum(theta_ref_mc(C2E2C), axis=C2E2CDim) / 3.0
    enh_diffu_3d = where(
        ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0.0),
        (thresh_tdiff - tdiff + trefdiff) * 5.0e-4,
        -1.7976931348623157e308,
    )
    return enh_diffu_3d


@program
def temporary_field_for_grid_point_cold_pools_enhancement(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    enh_diffu_3d: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
):
    _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v, theta_ref_mc, thresh_tdiff, out=enh_diffu_3d
    )
