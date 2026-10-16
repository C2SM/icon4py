# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, max_over, maximum, neighbor_sum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _cold_pools_enhanced_diffusion(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
) -> fa.CellKField[vpfloat]:
    theta_ref_mc_wp = astype(theta_ref_mc, wpfloat)

    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=dims.C2E2CDim) / wpfloat("3.0")
    trefdiff = theta_ref_mc_wp - astype(
        neighbor_sum(theta_ref_mc(C2E2C), axis=dims.C2E2CDim), wpfloat
    ) / wpfloat("3.0")
    enh_diffu_3d_vp = where(
        ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < wpfloat("0.0"))
        | (tdiff - trefdiff < wpfloat("1.5") * thresh_tdiff),
        astype((thresh_tdiff - tdiff + trefdiff) * wpfloat("5.0e-4"), vpfloat),
        smallest_vpfloat,
    )
    return enh_diffu_3d_vp


@gtx.field_operator
def _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    enh_diffu_3d = _cold_pools_enhanced_diffusion(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        smallest_vpfloat,
    )
    return maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=dims.E2CDim))
