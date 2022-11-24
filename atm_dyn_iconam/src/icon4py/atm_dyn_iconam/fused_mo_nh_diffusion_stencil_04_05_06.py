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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, neighbor_sum, maximum, where, int32

from icon4py.common.dimension import (
    E2C2V,
    E2ECV,
    ECVDim,
    EdgeDim,
    KDim,
    VertexDim,
)

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_04 import _mo_nh_diffusion_stencil_04
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_05 import _mo_nh_diffusion_stencil_05
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_06 import _mo_nh_diffusion_stencil_06

@field_operator
def _fused_mo_nh_diffusion_stencil_04_05_06(
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_v1: Field[[ECVDim], float],
    primal_normal_vert_v2: Field[[ECVDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    horz_idx: Field[[EdgeDim], int32],
    nudgezone_diff: float,
    fac_bdydiff_v: float,
    start_2nd_nudge_line_idx_e: int32,
) -> Field[[EdgeDim, KDim], float]:

    z_nabla4_e2 = _mo_nh_diffusion_stencil_04(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2, z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length)

    vn = where(
        horz_idx >= start_2nd_nudge_line_idx_e,
        _mo_nh_diffusion_stencil_05(area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, nudgecoeff_e, vn, nudgezone_diff),
        _mo_nh_diffusion_stencil_06(z_nabla2_e, area_edge, vn, fac_bdydiff_v)
    )

    return vn


@program
def fused_mo_nh_diffusion_stencil_04_05_06(
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
    primal_normal_vert_v1: Field[[ECVDim], float],
    primal_normal_vert_v2: Field[[ECVDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    horz_idx: Field[[EdgeDim], int32],
    nudgezone_diff: float,
    fac_bdydiff_v: float,
    start_2nd_nudge_line_idx_e: int32,
):
    _fused_mo_nh_diffusion_stencil_04_05_06(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2, z_nabla2_e, inv_vert_vert_length,
            inv_primal_edge_length, area_edge, kh_smag_e, diff_multfac_vn, nudgecoeff_e, vn, horz_idx, nudgezone_diff, fac_bdydiff_v,
            start_2nd_nudge_line_idx_e, out=vn)
