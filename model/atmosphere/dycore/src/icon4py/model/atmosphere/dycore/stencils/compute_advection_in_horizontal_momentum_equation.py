# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import maximum

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


@gtx.field_operator
def _compute_advection_in_horizontal_momentum_equation(
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
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
    levelmask: fa.KField[bool],
    nlev: gtx.int32,
    nrdmax: gtx.int32,
) -> fa.EdgeKField[ta.vpfloat]:
    upward_vorticity_at_vertices_on_model_levels = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)

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

    # k = broadcast(k, (dims.EdgeDim, dims.KDim))
    normal_wind_advective_tendency = concat_where(
        ((maximum(3, nrdmax - 2) - 1) <= dims.KDim) & (dims.KDim < (nlev - 4)),
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
            normal_wind_advective_tendency,
        ),
        normal_wind_advective_tendency,
    )
    return normal_wind_advective_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_horizontal_momentum_equation(
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
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
    levelmask: fa.KField[bool],
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """Formerly known as fused_velocity_advection_stencil_19_to_20."""

    _compute_advection_in_horizontal_momentum_equation(
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
        nlev,
        nrdmax,
        out=normal_wind_advective_tendency,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
