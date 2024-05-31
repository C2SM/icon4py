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
from gt4py.next.ffront.fbuiltins import int32, maximum, where

from icon4py.model.atmosphere.dycore.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.compute_advective_normal_wind_tendency import (
    _compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import (
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_19_to_20(
    vn: fa.EKwpField,
    geofac_rot: Field[[VertexDim, V2EDim], wpfloat],
    z_kin_hor_e: fa.EKvpField,
    coeff_gradekin: Field[[ECDim], vpfloat],
    z_ekinh: fa.CKvpField,
    vt: fa.EKvpField,
    f_e: fa.EwpField,
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CKvpField,
    vn_ie: fa.EKvpField,
    ddqz_z_full_e: fa.EKvpField,
    levelmask: Field[[KDim], bool],
    area_edge: fa.EwpField,
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    k: fa.KintField,
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    extra_diffu: bool,
    nlev: int32,
    nrdmax: int32,
) -> fa.EKvpField:
    zeta = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)

    ddt_vn_apc = _compute_advective_normal_wind_tendency(
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
    )

    ddt_vn_apc = (
        where(
            maximum(2, nrdmax - 2) <= k < nlev - 3,
            _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
                levelmask,
                c_lin_e,
                z_w_con_c_full,
                ddqz_z_full_e,
                area_edge,
                tangent_orientation,
                inv_primal_edge_length,
                zeta,
                geofac_grdiv,
                vn,
                ddt_vn_apc,
                cfl_w_limit,
                scalfac_exdiff,
                d_time,
            ),
            ddt_vn_apc,
        )
        if extra_diffu
        else ddt_vn_apc
    )

    return ddt_vn_apc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def fused_velocity_advection_stencil_19_to_20(
    vn: fa.EKwpField,
    geofac_rot: Field[[VertexDim, V2EDim], wpfloat],
    z_kin_hor_e: fa.EKvpField,
    coeff_gradekin: Field[[ECDim], vpfloat],
    z_ekinh: fa.CKvpField,
    vt: fa.EKvpField,
    f_e: fa.EwpField,
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CKvpField,
    vn_ie: fa.EKvpField,
    ddqz_z_full_e: fa.EKvpField,
    levelmask: Field[[KDim], bool],
    area_edge: fa.EwpField,
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    ddt_vn_apc: fa.EKvpField,
    k: fa.KintField,
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    extra_diffu: bool,
    nlev: int32,
    nrdmax: int32,
):
    _fused_velocity_advection_stencil_19_to_20(
        vn,
        geofac_rot,
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        levelmask,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        geofac_grdiv,
        k,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        extra_diffu,
        nlev,
        nrdmax,
        out=ddt_vn_apc,
    )
