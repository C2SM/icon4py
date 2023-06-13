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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.atm_dyn_iconam.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    _enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.atm_dyn_iconam.temporary_field_for_grid_point_cold_pools_enhancement import (
    _temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim


@field_operator
def _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
    kh_smag_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    enh_diffu_3d = _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v, theta_ref_mc, thresh_tdiff
    )
    kh_smag_e = _enhance_diffusion_coefficient_for_grid_point_cold_pools(
        kh_smag_e, enh_diffu_3d
    )
    return kh_smag_e


@program
def calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
    kh_smag_e: Field[[EdgeDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        kh_smag_e,
        out=kh_smag_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
