# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_vp import (
    _interpolate_edge_field_to_half_levels_vp,
)


@gtx.field_operator
def _compute_vt_vn_on_half_levels_and_kinetic_energy(
    vn: fa.EdgeKField[ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    nlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    vn_on_half_levels, horizontal_kinetic_energy_at_edges_on_model_levels = concat_where(
        1 <= dims.KDim < nlev,
        _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
            wgtfac_e, vn, tangential_wind
        ),
        (vn_on_half_levels, horizontal_kinetic_energy_at_edges_on_model_levels),
    )

    tangential_wind_on_half_levels = (
        concat_where(
            1 <= dims.KDim < nlev,
            _interpolate_edge_field_to_half_levels_vp(wgtfac_e, tangential_wind),
            tangential_wind_on_half_levels,
        )
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = concat_where(
        dims.KDim == 0,
        _compute_horizontal_kinetic_energy(vn, tangential_wind),
        (
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
        ),
    )

    return (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    )


@gtx.field_operator
def _compute_derived_horizontal_winds_and_kinetic_energy_and_contravariant_correction(
    vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    nflatlev: gtx.int32,
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    nlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    tangential_wind = concat_where(
        dims.KDim < nlev,
        _compute_tangential_wind(vn, rbf_vec_coeff_e),
        tangential_wind,
    )

    (
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
    ) = _compute_vt_vn_on_half_levels_and_kinetic_energy(
        vn,
        wgtfac_e,
        tangential_wind_on_half_levels,
        tangential_wind,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        nlev,
        skip_compute_predictor_vertical_advection,
    )

    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim < nlev,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    return (
        tangential_wind,
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )


@gtx.field_operator
def _compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    edge: fa.EdgeField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    lateral_boundary_7: gtx.int32,
    halo_1: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    (
        tangential_wind,
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    ) = _compute_derived_horizontal_winds_and_kinetic_energy_and_contravariant_correction(
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        contravariant_correction_at_edges_on_model_levels,
        nflatlev,
        tangential_wind_on_half_levels,
        tangential_wind,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        nlev,
        skip_compute_predictor_vertical_advection,
    )

    w_at_vertices = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)

    horizontal_advection_of_w_at_edges_on_half_levels = (
        concat_where(
            dims.KDim < nlev,
            where(
                lateral_boundary_7 <= edge < halo_1,
                _compute_horizontal_advection_term_for_vertical_velocity(
                    vn_on_half_levels,
                    inv_dual_edge_length,
                    w,
                    tangential_wind_on_half_levels,
                    inv_primal_edge_length,
                    tangent_orientation,
                    w_at_vertices,
                ),
                horizontal_advection_of_w_at_edges_on_half_levels,
            ),
            horizontal_advection_of_w_at_edges_on_half_levels,
        )
        if not skip_compute_predictor_vertical_advection
        else horizontal_advection_of_w_at_edges_on_half_levels
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
    )


@gtx.field_operator
def _compute_horizontal_advection_of_w(
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    edge: fa.EdgeField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    start_edge_lateral_boundary_level_7: gtx.int32,
    end_edge_halo: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
) -> fa.EdgeKField[ta.vpfloat]:
    w_at_vertices = where(
        (start_vertex_lateral_boundary_level_2 <= vertex < end_vertex_halo),
        _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp),
        0.0,
    )

    horizontal_advection_of_w_at_edges_on_half_levels = where(
        (start_edge_lateral_boundary_level_7 <= edge < end_edge_halo),
        _compute_horizontal_advection_term_for_vertical_velocity(
            vn_on_half_levels,
            inv_dual_edge_length,
            w,
            tangential_wind_on_half_levels,
            inv_primal_edge_length,
            tangent_orientation,
            w_at_vertices,
        ),
        horizontal_advection_of_w_at_edges_on_half_levels,
    )

    return horizontal_advection_of_w_at_edges_on_half_levels


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    edge: fa.EdgeField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    lateral_boundary_7: gtx.int32,
    halo_1: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """Formerly known as fused_velocity_advection_stencil_1_to_7_predictor."""
    _compute_derived_horizontal_winds_and_ke_and_horizontal_advection_of_w_and_contravariant_correction(
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
        vn,
        w,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        c_intp,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        skip_compute_predictor_vertical_advection,
        edge,
        nflatlev,
        nlev,
        lateral_boundary_7,
        halo_1,
        out=(
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_advection_of_w(
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    edge: fa.EdgeField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    lateral_boundary_7: gtx.int32,
    halo_1: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """Formerly known as fused_velocity_advection_stencil_1_to_7_corrector."""

    _compute_horizontal_advection_of_w(
        horizontal_advection_of_w_at_edges_on_half_levels,
        w,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        c_intp,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        edge,
        vertex,
        lateral_boundary_7,
        halo_1,
        start_vertex_lateral_boundary_level_2,
        end_vertex_halo,
        out=horizontal_advection_of_w_at_edges_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
