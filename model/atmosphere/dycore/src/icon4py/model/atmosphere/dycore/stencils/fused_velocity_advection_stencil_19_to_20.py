# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import maximum, where

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advective_normal_wind_tendency import (
    _compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_velocity_advection_stencil_19_to_20(
    vn: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    f_e: fa.EdgeField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    levelmask: fa.KField[bool],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    k: fa.KField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    edge: fa.EdgeField[gtx.int32],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    extra_diffu: bool,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    ddt_vn_apc: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    zeta = where(
        start_vertex_lateral_boundary_level_2 <= vertex < end_vertex_halo,
        _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot),
        0.0,
    )

    ddt_vn_apc = where(
        start_edge_nudging_level_2 <= edge < end_edge_local,
        _compute_advective_normal_wind_tendency(
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
        ),
        ddt_vn_apc,
    )

    k = broadcast(k, (dims.EdgeDim, dims.KDim))
    ddt_vn_apc = where(
        (start_edge_nudging_level_2 <= edge < end_edge_local)
        & ((maximum(3, nrdmax - 2) - 1) <= k < nlev - 4),
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
    # if extra_diffu
    # else ddt_vn_apc
    return ddt_vn_apc


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_19_to_20(
    vn: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    z_ekinh: fa.CellKField[vpfloat],
    vt: fa.EdgeKField[vpfloat],
    f_e: fa.EdgeField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    levelmask: fa.KField[bool],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    ddt_vn_apc: fa.EdgeKField[vpfloat],
    k: fa.KField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    edge: fa.EdgeField[gtx.int32],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    extra_diffu: bool,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
        vertex,
        edge,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        extra_diffu,
        nlev,
        nrdmax,
        start_vertex_lateral_boundary_level_2,
        end_vertex_halo,
        start_edge_nudging_level_2,
        end_edge_local,
        ddt_vn_apc,
        out=ddt_vn_apc,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
