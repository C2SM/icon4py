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
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_advection_in_horizontal_momentum_equation(
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[vpfloat],
    tangential_wind: fa.EdgeKField[vpfloat],
    coriolis_frequency: fa.EdgeField[wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[vpfloat],
    vn_on_half_levels: fa.EdgeKField[vpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    levelmask: fa.KField[bool],
    k: fa.KField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    edge: fa.EdgeField[gtx.int32],
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
) -> fa.EdgeKField[vpfloat]:
    upward_vorticity_at_vertices_on_model_levels = where(
        start_vertex_lateral_boundary_level_2 <= vertex < end_vertex_halo,
        _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot),
        0.0,
    )

    normal_wind_advective_tendency = where(
        start_edge_nudging_level_2 <= edge < end_edge_local,
        _compute_advective_normal_wind_tendency(
            horizontal_kinetic_energy_at_edges_on_model_levels,
            coeff_gradekin,
            horizontal_kinetic_energy_at_cells_on_model_levels,
            upward_vorticity_at_vertices_on_model_levels,
            tangential_wind,
            coriolis_frequency,
            c_lin_e,
            contravariant_corrected_w_at_cells_on_model_levels,
            vn_on_half_levels,
            ddqz_z_full_e,
        ),
        normal_wind_advective_tendency,
    )

    k = broadcast(k, (dims.EdgeDim, dims.KDim))
    normal_wind_advective_tendency = where(
        (start_edge_nudging_level_2 <= edge < end_edge_local)
        & ((maximum(3, nrdmax - 2) - 1) <= k < nlev - 4),
        _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
            levelmask,
            c_lin_e,
            contravariant_corrected_w_at_cells_on_model_levels,
            ddqz_z_full_e,
            area_edge,
            tangent_orientation,
            inv_primal_edge_length,
            upward_vorticity_at_vertices_on_model_levels,
            geofac_grdiv,
            vn,
            normal_wind_advective_tendency,
            cfl_w_limit,
            scalfac_exdiff,
            d_time,
        ),
        normal_wind_advective_tendency,
    )
    return normal_wind_advective_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_horizontal_momentum_equation(
    normal_wind_advective_tendency: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[vpfloat],
    tangential_wind: fa.EdgeKField[vpfloat],
    coriolis_frequency: fa.EdgeField[wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[vpfloat],
    vn_on_half_levels: fa.EdgeKField[vpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    d_time: wpfloat,
    levelmask: fa.KField[bool],
    k: fa.KField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    edge: fa.EdgeField[gtx.int32],
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
    _compute_advection_in_horizontal_momentum_equation(
        normal_wind_advective_tendency,
        vn,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        horizontal_kinetic_energy_at_cells_on_model_levels,
        tangential_wind,
        coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels,
        geofac_rot,
        coeff_gradekin,
        c_lin_e,
        ddqz_z_full_e,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        geofac_grdiv,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        levelmask,
        k,
        vertex,
        edge,
        nlev,
        nrdmax,
        start_vertex_lateral_boundary_level_2,
        end_vertex_halo,
        start_edge_nudging_level_2,
        end_edge_local,
        out=normal_wind_advective_tendency,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
