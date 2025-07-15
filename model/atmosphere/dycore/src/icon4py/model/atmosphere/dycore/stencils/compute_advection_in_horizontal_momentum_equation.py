# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype, maximum, minimum, neighbor_sum, where

from icon4py.model.atmosphere.dycore.stencils.compute_advective_normal_wind_tendency import (
    _compute_advective_normal_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import (
    E2C,
    E2C2EO,
    E2V,
    E2C2EODim,
    E2CDim,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    z_w_con_c_full: fa.CellKField[ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    zeta: fa.VertexKField[ta.vpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    ddt_vn_apc: fa.EdgeKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> fa.EdgeKField[ta.vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_20."""
    z_w_con_c_full_wp, ddqz_z_full_e_wp, ddt_vn_apc_wp, cfl_w_limit_wp = astype(
        (z_w_con_c_full, ddqz_z_full_e, ddt_vn_apc, cfl_w_limit), wpfloat
    )

    w_con_e = neighbor_sum(c_lin_e * z_w_con_c_full_wp(E2C), axis=E2CDim)
    difcoef = scalfac_exdiff * minimum(
        wpfloat("0.85") - cfl_w_limit_wp * dtime,
        abs(w_con_e) * dtime / ddqz_z_full_e_wp - cfl_w_limit_wp * dtime,
    )
    ddt_vn_apc_wp = where(
        abs(w_con_e) > astype(cfl_w_limit * ddqz_z_full_e, wpfloat),
        ddt_vn_apc_wp
        + difcoef
        * area_edge
        * (
            neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)
            + tangent_orientation
            * inv_primal_edge_length
            * astype(zeta(E2V[1]) - zeta(E2V[0]), wpfloat)
        ),
        ddt_vn_apc_wp,
    )
    return astype(ddt_vn_apc_wp, vpfloat)


@gtx.field_operator
def _compute_advection_in_horizontal_momentum_equation(
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    d_time: ta.wpfloat,
    max_vertical_cfl: fa.KField[bool],
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> fa.EdgeKField[ta.vpfloat]:
    upward_vorticity_at_vertices_on_model_levels = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)

    horizontal_kinetic_energy_at_cells_on_model_levels = _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
    )

    normal_wind_advective_tendency = _compute_advective_normal_wind_tendency(
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
    )

    if max_vertical_cfl > cfl_w_limit * d_time:
        normal_wind_advective_tendency = concat_where(
            ((maximum(3, end_index_of_damping_layer - 2) - 1) <= dims.KDim)
            & (dims.KDim < (nlev - 4)),
            _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
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
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.ECDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    d_time: ta.wpfloat,
    max_vertical_cfl: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_velocity_advection_stencil_19_to_20.

    This computes the horizontal advection in the horizontal momentum equation

    Args:
        - normal_wind_advective_tendency: horizontal advection tendency of the normal wind
        - vn: normal wind at edges
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy at edges on model levels
        - horizontal_kinetic_energy_at_cells_on_model_levels: horizontal kinetic energy at cell centers on model levels
        - tangential_wind: tangential wind at model levels
        - coriolis_frequency: coriolis frequency parameter
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at model levels
        - vn_on_half_levels: normal wind on half levels
        - geofac_rot: metric field for rotor computation
        - coeff_gradekin: metrics field/coefficient for the gradient of kinematic energy
        - c_lin_e: metrics field for linear interpolation from cells to edges
        - ddqz_z_full_e: metrics field equal to vertical spacing
        - area_edge: area associated with each edge
        - tangent_orientation: orientation of the edge with respect to the grid
        - inv_primal_edge_length: inverse primal edge length
        - geofac_grdiv: metrics field used to compute the gradient of a divergence (of vn)
        - cfl_w_limit: CFL limit for vertical velocity
        - scalfac_exdiff: scalar factor for external diffusion
        - d_time: time step
        - nlev: number of (full/model) vertical levels
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal domain
        - horizontal_end: end index in the horizontal domain
        - vertical_start: start index in the vertical domain
        - vertical_end: end index in the vertical domain

    Returns:
        - normal_wind_advective_tendency: horizontal advection tendency of the normal wind

    """

    _compute_advection_in_horizontal_momentum_equation(
        vn,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        tangential_wind,
        coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels,
        e_bln_c_s,
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
        max_vertical_cfl,
        nlev,
        end_index_of_damping_layer,
        out=normal_wind_advective_tendency,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
