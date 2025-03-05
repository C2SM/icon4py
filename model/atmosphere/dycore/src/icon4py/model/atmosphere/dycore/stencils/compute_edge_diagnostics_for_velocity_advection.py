# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.fbuiltins import broadcast, where

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
from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_ie_and_compute_ekin_on_edges import (
    _interpolate_vn_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vt_to_interface_edges import (
    _interpolate_vt_to_interface_edges,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO (Chia RUi): Rename and clean up the individual stencils used by the combined stencil
@field_operator
def _compute_interface_vt_vn_and_kinetic_energy(
    vn: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKField[vpfloat],
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    tangential_wind: fa.EdgeKField[vpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    khalf_vn, horizontal_kinetic_energy_at_edge = where(
        1 <= k < nlev,
        _interpolate_vn_to_ie_and_compute_ekin_on_edges(wgtfac_e, vn, tangential_wind),
        (khalf_vn, horizontal_kinetic_energy_at_edge),
    )

    khalf_tangential_wind = (
        where(
            1 <= k < nlev,
            _interpolate_vt_to_interface_edges(wgtfac_e, tangential_wind),
            khalf_tangential_wind,
        )
        if not skip_compute_predictor_vertical_advection
        else khalf_tangential_wind
    )

    (khalf_vn, khalf_tangential_wind, horizontal_kinetic_energy_at_edge) = where(
        k == 0,
        _compute_horizontal_kinetic_energy(vn, tangential_wind),
        (khalf_vn, khalf_tangential_wind, horizontal_kinetic_energy_at_edge),
    )

    return khalf_vn, khalf_tangential_wind, horizontal_kinetic_energy_at_edge


@field_operator
def _fused_velocity_advection_stencil_1_to_6(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    wgtfac_e: fa.EdgeKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    contravariant_correction_at_edge: fa.EdgeKField[vpfloat],
    nflatlev: gtx.int32,
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    tangential_wind: fa.EdgeKField[vpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    skip_compute_predictor_vertical_advection: bool,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    tangential_wind = where(
        k < nlev,
        _compute_tangential_wind(vn, rbf_vec_coeff_e),
        tangential_wind,
    )

    (
        khalf_vn,
        khalf_tangential_wind,
        horizontal_kinetic_energy_at_edge,
    ) = _compute_interface_vt_vn_and_kinetic_energy(
        vn,
        wgtfac_e,
        khalf_tangential_wind,
        tangential_wind,
        khalf_vn,
        horizontal_kinetic_energy_at_edge,
        k,
        nlev,
        skip_compute_predictor_vertical_advection,
    )

    contravariant_correction_at_edge = where(
        nflatlev <= k < nlev,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edge,
    )

    return (
        tangential_wind,
        khalf_vn,
        khalf_tangential_wind,
        horizontal_kinetic_energy_at_edge,
        contravariant_correction_at_edge,
    )


@field_operator
def _compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind: fa.EdgeKField[vpfloat],
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    contravariant_correction_at_edge: fa.EdgeKField[vpfloat],
    khalf_horizontal_advection_of_w_at_edge: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    wgtfac_e: fa.EdgeKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    k: fa.KField[gtx.int32],
    edge: fa.EdgeField[gtx.int32],
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    lateral_boundary_7: gtx.int32,
    halo_1: gtx.int32,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    (
        tangential_wind,
        khalf_vn,
        khalf_tangential_wind,
        horizontal_kinetic_energy_at_edge,
        contravariant_correction_at_edge,
    ) = _fused_velocity_advection_stencil_1_to_6(
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        contravariant_correction_at_edge,
        nflatlev,
        khalf_tangential_wind,
        tangential_wind,
        khalf_vn,
        horizontal_kinetic_energy_at_edge,
        k,
        nlev,
        skip_compute_predictor_vertical_advection,
    )

    k = broadcast(k, (dims.EdgeDim, dims.KDim))

    khalf_w_at_vertex = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)

    khalf_horizontal_advection_of_w_at_edge = (
        where(
            (lateral_boundary_7 <= edge) & (edge < halo_1) & (k < nlev),
            _compute_horizontal_advection_term_for_vertical_velocity(
                khalf_vn,
                inv_dual_edge_length,
                w,
                khalf_tangential_wind,
                inv_primal_edge_length,
                tangent_orientation,
                khalf_w_at_vertex,
            ),
            khalf_horizontal_advection_of_w_at_edge,
        )
        if not skip_compute_predictor_vertical_advection
        else khalf_horizontal_advection_of_w_at_edge
    )

    return (
        tangential_wind,
        khalf_tangential_wind,
        khalf_vn,
        horizontal_kinetic_energy_at_edge,
        contravariant_correction_at_edge,
        khalf_horizontal_advection_of_w_at_edge,
    )


@field_operator
def _compute_khalf_horizontal_advection_of_w(
    khalf_horizontal_advection_of_w_at_edge: fa.EdgeKField[vpfloat],
    w: fa.CellKField[wpfloat],
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    edge: fa.EdgeField[gtx.int32],
    vertex: fa.VertexField[gtx.int32],
    start_edge_lateral_boundary_level_7: gtx.int32,
    end_edge_halo: gtx.int32,
    start_vertex_lateral_boundary_level_2: gtx.int32,
    end_vertex_halo: gtx.int32,
) -> fa.EdgeKField[vpfloat]:
    khalf_w_at_vertex = where(
        (start_vertex_lateral_boundary_level_2 <= vertex < end_vertex_halo),
        _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp),
        0.0,
    )

    khalf_horizontal_advection_of_w_at_edge = where(
        (start_edge_lateral_boundary_level_7 <= edge < end_edge_halo),
        _compute_horizontal_advection_term_for_vertical_velocity(
            khalf_vn,
            inv_dual_edge_length,
            w,
            khalf_tangential_wind,
            inv_primal_edge_length,
            tangent_orientation,
            khalf_w_at_vertex,
        ),
        khalf_horizontal_advection_of_w_at_edge,
    )

    return khalf_horizontal_advection_of_w_at_edge


@program(grid_type=GridType.UNSTRUCTURED)
def compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction(
    tangential_wind: fa.EdgeKField[vpfloat],
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    horizontal_kinetic_energy_at_edge: fa.EdgeKField[vpfloat],
    contravariant_correction_at_edge: fa.EdgeKField[vpfloat],
    khalf_horizontal_advection_of_w_at_edge: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    wgtfac_e: fa.EdgeKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    wgtfacq_e: fa.EdgeKField[vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    k: fa.KField[gtx.int32],
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
    _compute_vt_and_khalf_winds_and_horizontal_advection_of_w_and_contravariant_correction(
        tangential_wind,
        khalf_tangential_wind,
        khalf_vn,
        horizontal_kinetic_energy_at_edge,
        contravariant_correction_at_edge,
        khalf_horizontal_advection_of_w_at_edge,
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
        k,
        edge,
        nflatlev,
        nlev,
        lateral_boundary_7,
        halo_1,
        out=(
            tangential_wind,
            khalf_tangential_wind,
            khalf_vn,
            horizontal_kinetic_energy_at_edge,
            contravariant_correction_at_edge,
            khalf_horizontal_advection_of_w_at_edge,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=khalf_vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED)
def compute_khalf_horizontal_advection_of_w(
    khalf_horizontal_advection_of_w_at_edge: fa.EdgeKField[vpfloat],
    w: fa.CellKField[wpfloat],
    khalf_tangential_wind: fa.EdgeKField[wpfloat],
    khalf_vn: fa.EdgeKField[vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
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
    _compute_khalf_horizontal_advection_of_w(
        khalf_horizontal_advection_of_w_at_edge,
        w,
        khalf_tangential_wind,
        khalf_vn,
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
        out=khalf_horizontal_advection_of_w_at_edge,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
