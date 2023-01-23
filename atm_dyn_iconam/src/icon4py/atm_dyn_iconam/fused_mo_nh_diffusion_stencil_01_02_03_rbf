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
from functional.ffront.fbuiltins import Field

from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_01 import (
    _mo_nh_diffusion_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_02 import (
    _mo_nh_diffusion_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_03 import (
    _mo_nh_diffusion_stencil_03,
)
from icon4py.common.dimension import (
    C2EDim,
    CellDim,
    ECVDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)


@field_operator
def _fused_mo_nh_diffusion_stencil_01_02_03_rbf(
    diff_multfac_smag: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert_old: Field[[VertexDim, KDim], float],
    v_vert_old: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    smag_offset: float,
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    ptr_coeff_1: Field[[VertexDim, V2EDim], float],
    ptr_coeff_2: Field[[VertexDim, V2EDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[VertexDim, KDim], float],
    Field[[VertexDim, KDim], float],
]:

    kh_smag_e, kh_smag_ec, z_nabla2_e = _mo_nh_diffusion_stencil_01(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert_old,
        v_vert_old,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
    )

    kh_c, div = _mo_nh_diffusion_stencil_02(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )

    div_ic, hdef_ic = _mo_nh_diffusion_stencil_03(div, kh_c, wgtfac_c)

    u_vert, v_vert = _mo_intp_rbf_rbf_vec_interpol_vertex(
        z_nabla2_e, ptr_coeff_1, ptr_coeff_2
    )

    return kh_smag_e, div_ic, hdef_ic, u_vert, v_vert


@program
def fused_mo_nh_diffusion_stencil_01_02_03_rbf(
    diff_multfac_smag: Field[[KDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    inv_vert_vert_length: Field[[EdgeDim], float],
    u_vert_old: Field[[VertexDim, KDim], float],
    v_vert_old: Field[[VertexDim, KDim], float],
    primal_normal_vert_x: Field[[ECVDim], float],
    primal_normal_vert_y: Field[[ECVDim], float],
    dual_normal_vert_x: Field[[ECVDim], float],
    dual_normal_vert_y: Field[[ECVDim], float],
    vn: Field[[EdgeDim, KDim], float],
    smag_limit: Field[[KDim], float],
    smag_offset: float,
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    ptr_coeff_1: Field[[VertexDim, V2EDim], float],
    ptr_coeff_2: Field[[VertexDim, V2EDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    u_vert: Field[[VertexDim, KDim], float],
    v_vert: Field[[VertexDim, KDim], float],
):
    _fused_mo_nh_diffusion_stencil_01_02_03_rbf(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert_old,
        v_vert_old,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
        e_bln_c_s,
        geofac_div,
        wgtfac_c,
        ptr_coeff_1,
        ptr_coeff_2,
        out=(kh_smag_e, div_ic, hdef_ic, u_vert, v_vert),
    )
