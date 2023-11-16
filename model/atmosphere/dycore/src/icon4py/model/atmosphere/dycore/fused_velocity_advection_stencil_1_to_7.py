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
from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_01 import (
    _mo_velocity_advection_stencil_01,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_02 import (
    _mo_velocity_advection_stencil_02,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_03 import (
    _mo_velocity_advection_stencil_03,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_04 import (
    _mo_velocity_advection_stencil_04,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_05 import (
    _mo_velocity_advection_stencil_05,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_06 import (
    _mo_velocity_advection_stencil_06,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_07 import (
    _mo_velocity_advection_stencil_07,
)
from icon4py.model.common.dimension import CellDim, E2C2EDim, EdgeDim, KDim, V2CDim, VertexDim


@field_operator
def _fused_velocity_advection_stencil_1_to_6(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    nflatlev: int32,
    z_vt_ie: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    k: Field[[KDim], int32],
    nlev: int32,
    lvn_only: bool,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    vt = where(
        k < nlev,
        _mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e),
        vt,
    )

    vn_ie, z_kin_hor_e = where(
        1 <= k < nlev,
        _mo_velocity_advection_stencil_02(wgtfac_e, vn, vt),
        (vn_ie, z_kin_hor_e),
    )

    z_vt_ie = (
        where(
            1 <= k < nlev,
            _mo_velocity_advection_stencil_03(wgtfac_e, vt),
            z_vt_ie,
        )
        if not lvn_only
        else z_vt_ie
    )

    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k == int32(0),
        _mo_velocity_advection_stencil_05(vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )

    vn_ie = where(k == nlev, _mo_velocity_advection_stencil_06(wgtfacq_e, vn), vn_ie)

    z_w_concorr_me = where(
        nflatlev <= k < nlev,
        _mo_velocity_advection_stencil_04(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )

    return vt, vn_ie, z_kin_hor_e, z_w_concorr_me


@field_operator
def _fused_velocity_advection_stencil_1_to_7(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    nflatlev: int32,
    c_intp: Field[[VertexDim, V2CDim], float],
    w: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    lvn_only: bool,
    edge: Field[[EdgeDim], int32],
    lateral_boundary_7: int32,
    halo_1: int32,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    vt, vn_ie, z_kin_hor_e, z_w_concorr_me = (
        _fused_velocity_advection_stencil_1_to_6(
            vn,
            rbf_vec_coeff_e,
            wgtfac_e,
            ddxn_z_full,
            ddxt_z_full,
            z_w_concorr_me,
            wgtfacq_e,
            nflatlev,
            z_vt_ie,
            vt,
            vn_ie,
            z_kin_hor_e,
            k,
            nlev,
            lvn_only,
        )
        if istep == 1
        else (vt, vn_ie, z_kin_hor_e, z_w_concorr_me)
    )

    k = broadcast(k, (EdgeDim, KDim))

    z_w_v = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)

    z_v_grad_w = (
        where(
            (lateral_boundary_7 <= edge < halo_1) & (k < nlev),
            _mo_velocity_advection_stencil_07(
                vn_ie,
                inv_dual_edge_length,
                w,
                z_vt_ie,
                inv_primal_edge_length,
                tangent_orientation,
                z_w_v,
            ),
            z_v_grad_w,
        )
        if not lvn_only
        else z_v_grad_w
    )

    return vt, vn_ie, z_kin_hor_e, z_w_concorr_me, z_v_grad_w

@field_operator
def _fused_velocity_advection_stencil_1_to_7_restricted(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    nflatlev: int32,
    c_intp: Field[[VertexDim, V2CDim], float],
    w: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    lvn_only: bool,
    edge: Field[[EdgeDim], int32],
    lateral_boundary_7: int32,
    halo_1: int32,
) -> Field[[EdgeDim, KDim], float]:


    return _fused_velocity_advection_stencil_1_to_7(
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        nflatlev,
        c_intp,
        w,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        z_v_grad_w,
        k,
        istep,
        nlev,
        lvn_only,
        edge,
        lateral_boundary_7,
        halo_1,
    )[1]


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_1_to_7(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    nflatlev: int32,
    c_intp: Field[[VertexDim, V2CDim], float],
    w: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    lvn_only: bool,
    edge: Field[[EdgeDim], int32],
    lateral_boundary_7: int32,
    halo_1: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_velocity_advection_stencil_1_to_7(
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        nflatlev,
        c_intp,
        w,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        z_v_grad_w,
        k,
        istep,
        nlev,
        lvn_only,
        edge,
        lateral_boundary_7,
        halo_1,
        out=(vt, vn_ie, z_kin_hor_e, z_w_concorr_me, z_v_grad_w),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
    _fused_velocity_advection_stencil_1_to_7_restricted(
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e,
        nflatlev,
        c_intp,
        w,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        z_v_grad_w,
        k,
        istep,
        nlev,
        lvn_only,
        edge,
        lateral_boundary_7,
        halo_1,
        out=vn_ie,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )
