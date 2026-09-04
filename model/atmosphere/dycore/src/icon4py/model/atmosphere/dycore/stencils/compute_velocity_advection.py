# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import (
    abs,  # noqa: A004
    astype,
    broadcast,
    maximum,
    minimum,
    neighbor_sum,
    where,
)
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    _add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    _compute_horizontal_kinetic_energy,
    _interpolate_to_half_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2EO, E2V
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center_vp import (
    _interpolate_to_cell_center_vp,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_diagnostics_from_normal_wind(
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    tangential_wind = _compute_tangential_wind_vp(vn, rbf_vec_coeff_e)
    horizontal_kinetic_energy_at_edges_on_model_levels = _compute_horizontal_kinetic_energy(
        vn, tangential_wind
    )
    vn_on_half_levels = concat_where(
        dims.KHalfDim < nlev,
        _interpolate_to_half_levels(wgtfac_e, vn),
        _extrapolate_at_top(wgtfacq_e, vn),
    )

    tangential_wind_on_half_levels = (
        _interpolate_to_half_levels(wgtfac_e, tangential_wind)
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    contravariant_correction_at_edges_on_model_levels = _compute_contravariant_correction(
        vn, ddxn_z_full, ddxt_z_full, tangential_wind
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )


@gtx.field_operator
def _interpolate_contravariant_vertical_velocity_to_full_levels(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    # TODO(havogt): Note that `concat_where(dims.KDim == nlev-1, ...)` is currently broken
    # because of insufficiency in the domain inference of GT4Py,
    # see https://github.com/GridTools/gt4py/issues/2205.
    return concat_where(
        dims.KDim < nlev - 1,
        vpfloat("0.5")
        * (
            contravariant_corrected_w_at_cells_on_half_levels(dims.KDim - 0.5)
            + contravariant_corrected_w_at_cells_on_half_levels(dims.KDim + 0.5)
        ),
        vpfloat("0.5") * contravariant_corrected_w_at_cells_on_half_levels(dims.KDim - 0.5),
    )


@gtx.field_operator
def _compute_horizontal_advection_of_w(
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKHalfField[ta.vpfloat]:
    w_at_vertices = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)

    horizontal_advection_of_w_at_edges_on_half_levels = (
        _compute_horizontal_advection_term_for_vertical_velocity(
            vn_on_half_levels,
            inv_dual_edge_length,
            w,
            tangential_wind_on_half_levels,
            inv_primal_edge_length,
            tangent_orientation,
            w_at_vertices,
        )
    )

    return astype(horizontal_advection_of_w_at_edges_on_half_levels, vpfloat)


@gtx.field_operator
def _add_vertical_advection_of_w_to_advective_vertical_wind_tendency(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    w: fa.CellKHalfField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
) -> fa.CellKHalfField[vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels_wp = astype(
        contravariant_corrected_w_at_cells_on_half_levels, wpfloat
    )
    coeff1_dwdz_at_half_levels = coeff1_dwdz(dims.KHalfDim + 0.5)
    coeff2_dwdz_at_half_levels = coeff2_dwdz(dims.KHalfDim + 0.5)
    coeff1_dwdz_wp, coeff2_dwdz_wp = astype(
        (coeff1_dwdz_at_half_levels, coeff2_dwdz_at_half_levels), wpfloat
    )

    vertical_wind_advective_tendency_wp = -contravariant_corrected_w_at_cells_on_half_levels_wp * (
        w(dims.KHalfDim - 1) * coeff1_dwdz_wp
        - w(dims.KHalfDim + 1) * coeff2_dwdz_wp
        + w * astype(coeff2_dwdz_at_half_levels - coeff1_dwdz_at_half_levels, wpfloat)
    )
    return astype(vertical_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> tuple[
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[bool],
    fa.CellKHalfField[ta.vpfloat],
]:
    contravariant_corrected_w_at_cells_on_half_levels_wp, ddqz_z_half_wp = astype(
        (contravariant_corrected_w_at_cells_on_half_levels, ddqz_z_half), wpfloat
    )

    cfl_clipping = where(
        abs(contravariant_corrected_w_at_cells_on_half_levels) > cfl_w_limit * ddqz_z_half,
        broadcast(True, (dims.CellDim, dims.KHalfDim)),
        False,
    )

    vertical_cfl = where(
        cfl_clipping,
        contravariant_corrected_w_at_cells_on_half_levels_wp * dtime / ddqz_z_half_wp,
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KHalfDim)),
    )
    vertical_cfl_vp = astype(vertical_cfl, vpfloat)

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl_vp < -vpfloat("0.85")),
        astype(-vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl_vp > vpfloat("0.85")),
        astype(vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    return (
        astype(contravariant_corrected_w_at_cells_on_half_levels_wp, vpfloat),
        cfl_clipping,
        vertical_cfl_vp,
    )


@gtx.field_operator
def _compute_contravariant_corrected_w(
    w: fa.CellKHalfField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
) -> fa.CellKHalfField[ta.vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels = (
        astype(w, vpfloat) - contravariant_correction_at_cells_on_half_levels
    )

    return contravariant_corrected_w_at_cells_on_half_levels


@gtx.field_operator
def _compute_contravariant_corrected_w_and_cfl(
    w: fa.CellKHalfField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKHalfField[ta.vpfloat], fa.CellKHalfField[bool], fa.CellKHalfField[ta.vpfloat]]:
    #: intermediate variable contravariant_corrected_w_at_cells_on_half_levels is originally declared as z_w_con_c in ICON
    contravariant_corrected_w_at_cells_on_half_levels = _compute_contravariant_corrected_w(
        w, contravariant_correction_at_cells_on_half_levels
    )

    (contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl) = concat_where(
        (dims.KHalfDim >= maximum(2, end_index_of_damping_layer - 2)) & (dims.KHalfDim < nlev - 3),
        _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            ddqz_z_half=ddqz_z_half,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        ),
        (
            contravariant_corrected_w_at_cells_on_half_levels,
            broadcast(False, (dims.CellDim, dims.KHalfDim)),
            broadcast(vpfloat("0.0"), (dims.CellDim, dims.KHalfDim)),
        ),
    )

    return contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl


@gtx.field_operator
def _compute_advective_vertical_wind_tendency(
    w: fa.CellKHalfField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[ta.wpfloat],
    cfl_clipping: fa.CellKHalfField[bool],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.vpfloat]:
    vertical_wind_advective_tendency = (
        _add_vertical_advection_of_w_to_advective_vertical_wind_tendency(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        )
    )

    vertical_wind_advective_tendency = _add_interpolated_horizontal_advection_of_w(
        e_bln_c_s,
        horizontal_advection_of_w_at_edges_on_half_levels,
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = _add_extra_diffusion_for_w_con_approaching_cfl(
        cfl_clipping,
        owner_mask,
        contravariant_corrected_w_at_cells_on_half_levels,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        vertical_wind_advective_tendency,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
    )

    return vertical_wind_advective_tendency


@gtx.field_operator
def _compute_advection_in_corrector_vertical_momentum(
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKHalfField[ta.vpfloat], fa.CellKField[ta.vpfloat], fa.CellKHalfField[ta.vpfloat]]:
    #: intermediate variable horizontal_advection_of_w_at_edges_on_half_levels is originally declared as z_v_grad_w in ICON
    horizontal_advection_of_w_at_edges_on_half_levels = _compute_horizontal_advection_of_w(
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
    )

    (
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
    ) = _compute_contravariant_corrected_w_and_cfl(
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    vertical_wind_advective_tendency = _compute_advective_vertical_wind_tendency(
        w=w,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping=cfl_clipping,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
    )

    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels, nlev
        )
    )

    return (
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    )


@gtx.field_operator
def _interpolate_contravariant_correction_to_cells_on_half_levels(
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    nflatlev: gtx.int32,
) -> fa.CellKHalfField[ta.vpfloat]:
    contravariant_correction_at_cells_model_levels = _interpolate_to_cell_center_vp(
        contravariant_correction_at_edges_on_model_levels, e_bln_c_s
    )
    contravariant_correction_at_cells_model_levels = astype(
        contravariant_correction_at_cells_model_levels, vpfloat
    )

    contravariant_correction_at_cells_on_half_levels = concat_where(
        dims.KHalfDim >= nflatlev + 1,
        _interpolate_cell_field_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        broadcast(vpfloat("0.0"), (dims.CellDim, dims.KHalfDim)),
    )

    return contravariant_correction_at_cells_on_half_levels


@gtx.field_operator
def _compute_advection_in_predictor_vertical_momentum(
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
]:
    contravariant_correction_at_cells_on_half_levels = _interpolate_contravariant_correction_to_cells_on_half_levels(
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        nflatlev=nflatlev,
    )

    (
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
    ) = _compute_contravariant_corrected_w_and_cfl(
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    if not skip_compute_predictor_vertical_advection:
        #: intermediate variable horizontal_advection_of_w_at_edges_on_half_levels is originally declared as z_v_grad_w in ICON
        horizontal_advection_of_w_at_edges_on_half_levels = _compute_horizontal_advection_of_w(
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
        )
        vertical_wind_advective_tendency = _compute_advective_vertical_wind_tendency(
            w=w,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=cfl_clipping,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        )

    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels, nlev
        )
    )

    return (
        contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    )


@gtx.field_operator
def _compute_advective_normal_wind_tendency(
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    #: intermediate variable horizontal_kinetic_energy_at_cells_on_model_levels is originally declared as z_ekinh in ICON
    horizontal_kinetic_energy_at_cells_on_model_levels = _interpolate_to_cell_center_vp(
        horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
    )
    horizontal_kinetic_energy_at_cells_on_model_levels = astype(
        horizontal_kinetic_energy_at_cells_on_model_levels, vpfloat
    )

    (
        contravariant_corrected_w_at_cells_on_model_levels_wp,
        ddqz_z_full_e_wp,
        tangential_wind_wp,
    ) = astype(
        (contravariant_corrected_w_at_cells_on_model_levels, ddqz_z_full_e, tangential_wind),
        wpfloat,
    )

    horizontal_advection = (
        horizontal_kinetic_energy_at_edges_on_model_levels
        * (coeff_gradekin[dims.E2CDim(0)] - coeff_gradekin[dims.E2CDim(1)])
        + coeff_gradekin[dims.E2CDim(1)]
        * horizontal_kinetic_energy_at_cells_on_model_levels(E2C[1])
        - coeff_gradekin[dims.E2CDim(0)]
        * horizontal_kinetic_energy_at_cells_on_model_levels(E2C[0])
    )

    vertical_advection = (
        neighbor_sum(
            c_lin_e * contravariant_corrected_w_at_cells_on_model_levels_wp(E2C), axis=dims.E2CDim
        )
        * astype((vn_on_half_levels(dims.KDim - 0.5) - vn_on_half_levels(dims.KDim + 0.5)), wpfloat)
        / ddqz_z_full_e_wp
    )

    coriolis_term = tangential_wind_wp * (
        coriolis_frequency
        + astype(
            vpfloat("0.5")
            * neighbor_sum(upward_vorticity_at_vertices_on_model_levels(E2V), axis=dims.E2VDim),
            wpfloat,
        )
    )
    normal_wind_advective_tendency_wp = -(horizontal_advection + vertical_advection + coriolis_term)

    return astype(normal_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_extra_diffusion(
    vn: fa.EdgeKField[ta.wpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    difcoef: fa.EdgeKField[ta.wpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    gradient_of_divergence_of_vn = neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=dims.E2C2EODim)

    gradient_of_vorticity = (
        tangent_orientation
        * inv_primal_edge_length
        * astype(
            upward_vorticity_at_vertices_on_model_levels(E2V[1])
            - upward_vorticity_at_vertices_on_model_levels(E2V[0]),
            wpfloat,
        )
    )

    extra_diffusion_on_vn = (
        difcoef * area_edge * (gradient_of_divergence_of_vn + gradient_of_vorticity)
    )

    return extra_diffusion_on_vn


@gtx.field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> fa.EdgeKField[ta.vpfloat]:
    (
        contravariant_corrected_w_at_cells_on_model_levels_wp,
        ddqz_z_full_e_wp,
        normal_wind_advective_tendency_wp,
        cfl_w_limit_wp,
    ) = astype(
        (
            contravariant_corrected_w_at_cells_on_model_levels,
            ddqz_z_full_e,
            normal_wind_advective_tendency,
            cfl_w_limit,
        ),
        wpfloat,
    )

    #: intermediate variable contravariant_corrected_w_at_edges_on_model_levels is originally declared as w_con_e in ICON
    contravariant_corrected_w_at_edges_on_model_levels = neighbor_sum(
        c_lin_e * contravariant_corrected_w_at_cells_on_model_levels_wp(E2C), axis=dims.E2CDim
    )
    difcoef = scalfac_exdiff * minimum(
        wpfloat("0.85") - cfl_w_limit_wp * dtime,
        abs(contravariant_corrected_w_at_edges_on_model_levels) * dtime / ddqz_z_full_e_wp
        - cfl_w_limit_wp * dtime,
    )
    normal_wind_advective_tendency_wp = where(
        abs(contravariant_corrected_w_at_edges_on_model_levels)
        > astype(cfl_w_limit * ddqz_z_full_e, wpfloat),
        normal_wind_advective_tendency_wp
        + _compute_extra_diffusion(
            vn=vn,
            upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
            difcoef=difcoef,
            area_edge=area_edge,
            geofac_grdiv=geofac_grdiv,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
        ),
        normal_wind_advective_tendency_wp,
    )
    return astype(normal_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_advection_in_horizontal_momentum(
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> fa.EdgeKField[ta.vpfloat]:
    upward_vorticity_at_vertices_on_model_levels = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)
    upward_vorticity_at_vertices_on_model_levels = astype(
        upward_vorticity_at_vertices_on_model_levels, vpfloat
    )

    normal_wind_advective_tendency = _compute_advective_normal_wind_tendency(
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
        tangential_wind=tangential_wind,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        coriolis_frequency=coriolis_frequency,
        e_bln_c_s=e_bln_c_s,
        c_lin_e=c_lin_e,
        coeff_gradekin=coeff_gradekin,
        ddqz_z_full_e=ddqz_z_full_e,
    )

    if apply_extra_diffusion_on_vn:
        normal_wind_advective_tendency = concat_where(
            ((maximum(2, end_index_of_damping_layer - 2)) <= dims.KDim) & (dims.KDim < (nlev - 4)),
            _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
                c_lin_e=c_lin_e,
                contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
                ddqz_z_full_e=ddqz_z_full_e,
                area_edge=area_edge,
                tangent_orientation=tangent_orientation,
                inv_primal_edge_length=inv_primal_edge_length,
                upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
                geofac_grdiv=geofac_grdiv,
                vn=vn,
                normal_wind_advective_tendency=normal_wind_advective_tendency,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                dtime=dtime,
            ),
            normal_wind_advective_tendency,
        )

    return normal_wind_advective_tendency


@gtx.field_operator
def _compute_velocity_advection_in_predictor_step(
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    apply_extra_diffusion_on_vn: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    ) = _compute_diagnostics_from_normal_wind(
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nlev=nlev,
    )

    (
        contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    ) = _compute_advection_in_predictor_vertical_momentum(
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nflatlev=nflatlev,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    normal_wind_advective_tendency = _compute_advection_in_horizontal_momentum(
        vn=vn,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        e_bln_c_s=e_bln_c_s,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        dtime=dtime,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency,
        vertical_cfl,
        normal_wind_advective_tendency,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_velocity_advection_in_predictor_step(
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    vertical_cfl: fa.CellKHalfField[ta.vpfloat],
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    apply_extra_diffusion_on_vn: bool,
    nflatlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_cell_lateral_boundary_level_4: gtx.int32,
    end_cell_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Compute the velocity advection tendencies of the predictor step.

    This is the port of ICON's `velocity_tendencies` (`mo_velocity_advection.f90`) for
    `istep == 1`: the wind quantities derived from the normal wind, the advection in the
    vertical momentum equation and the advection in the horizontal momentum equation.

    Args:
        - tangential_wind: tangential wind at edges on model levels
        - tangential_wind_on_half_levels: tangential wind at edges on half levels
        - vn_on_half_levels: normal wind at edges on half levels
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy at edges on model levels
        - contravariant_correction_at_edges_on_model_levels: contravariant metric correction at edges on model levels
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - vertical_wind_advective_tendency: advective tendency of the vertical wind
        - vertical_cfl: vertical cfl number at cells on half levels
        - normal_wind_advective_tendency: advective tendency of the normal wind
        - vn: normal wind at edges
        - w: vertical wind at cell centers
        - rbf_vec_coeff_e: interpolation field (RBF vector coefficient on edges)
        - wgtfac_e: metrics field
        - wgtfacq_e: metrics field (weights for interpolation)
        - ddxn_z_full: metrics field (derivative of topography in the normal direction)
        - ddxt_z_full: metrics field (derivative of topography in the tangential direction)
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - c_intp: interpolation field for cell-to-vertex interpolation
        - inv_dual_edge_length: inverse dual edge length
        - inv_primal_edge_length: inverse primal edge length
        - tangent_orientation: orientation of the edge with respect to the grid
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - wgtfac_c: metric coefficient for interpolating a cell variable from full to half levels
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - owner_mask: ownership mask for each cell
        - coriolis_frequency: coriolis frequency parameter
        - geofac_rot: metric field for rotor computation
        - coeff_gradekin: metrics field/coefficient for the gradient of kinematic energy
        - c_lin_e: metrics field for linear interpolation from cells to edges
        - ddqz_z_full_e: metrics field equal to vertical spacing
        - area_edge: area associated with each edge
        - geofac_grdiv: metrics field used to compute the gradient of a divergence (of vn)
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - skip_compute_predictor_vertical_advection: logical flag to skip the vertical advection
        - apply_extra_diffusion_on_vn: option to apply extra diffusion to vn
        - nflatlev: index of the first flat level
        - end_index_of_damping_layer: vertical index where damping ends
        - start_edge_lateral_boundary_level_5: start index of the edge quantities derived from vn
        - end_edge_halo_level_2: end index of the edge quantities derived from vn
        - start_cell_lateral_boundary_level_4: start index of the cell quantities
        - end_cell_halo: end index of the cell quantities
        - start_edge_nudging_level_2: start index of the normal wind tendency
        - end_edge_local: end index of the normal wind tendency
        - vertical_start: start index in the vertical dimension at model top
        - vertical_end: end index in the vertical dimension at model bottom (number of model levels)
    """

    _compute_velocity_advection_in_predictor_step(
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        vn=vn,
        w=w,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        coriolis_frequency=coriolis_frequency,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        geofac_grdiv=geofac_grdiv,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nflatlev=nflatlev,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=(
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency,
            vertical_cfl,
            normal_wind_advective_tendency,
        ),
        domain=(
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end + 1),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (nflatlev, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _compute_velocity_advection_in_corrector_step(
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    (
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    ) = _compute_advection_in_corrector_vertical_momentum(
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    normal_wind_advective_tendency = _compute_advection_in_horizontal_momentum(
        vn=vn,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        e_bln_c_s=e_bln_c_s,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        dtime=dtime,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    return (
        vertical_wind_advective_tendency,
        vertical_cfl,
        normal_wind_advective_tendency,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_velocity_advection_in_corrector_step(
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    vertical_cfl: fa.CellKHalfField[ta.vpfloat],
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    end_index_of_damping_layer: gtx.int32,
    start_cell_lateral_boundary_level_4: gtx.int32,
    end_cell_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Compute the velocity advection tendencies of the corrector step.

    This is the port of ICON's `velocity_tendencies` (`mo_velocity_advection.f90`) for
    `istep == 2`: the advection in the vertical momentum equation and the advection in the
    horizontal momentum equation. The wind quantities derived from the normal wind are
    computed in `compute_horizontal_velocity_quantities_and_fluxes` and passed in.

    Args:
        - vertical_wind_advective_tendency: advective tendency of the vertical wind
        - vertical_cfl: vertical cfl number at cells on half levels
        - normal_wind_advective_tendency: advective tendency of the normal wind
        - vn: normal wind at edges
        - w: vertical wind at cell centers
        - tangential_wind: tangential wind at edges on model levels
        - tangential_wind_on_half_levels: tangential wind at edges on half levels
        - vn_on_half_levels: normal wind at edges on half levels
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy at edges on model levels
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - c_intp: interpolation field for cell-to-vertex interpolation
        - inv_dual_edge_length: inverse dual edge length
        - inv_primal_edge_length: inverse primal edge length
        - tangent_orientation: orientation of the edge with respect to the grid
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - owner_mask: ownership mask for each cell
        - coriolis_frequency: coriolis frequency parameter
        - geofac_rot: metric field for rotor computation
        - coeff_gradekin: metrics field/coefficient for the gradient of kinematic energy
        - c_lin_e: metrics field for linear interpolation from cells to edges
        - ddqz_z_full_e: metrics field equal to vertical spacing
        - area_edge: area associated with each edge
        - geofac_grdiv: metrics field used to compute the gradient of a divergence (of vn)
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - apply_extra_diffusion_on_vn: option to apply extra diffusion to vn
        - end_index_of_damping_layer: vertical index where damping ends
        - start_cell_lateral_boundary_level_4: start index of the cell quantities
        - end_cell_halo: end index of the cell quantities
        - start_edge_nudging_level_2: start index of the normal wind tendency
        - end_edge_local: end index of the normal wind tendency
        - vertical_start: start index in the vertical dimension at model top
        - vertical_end: end index in the vertical dimension at model bottom (number of model levels)
    """

    _compute_velocity_advection_in_corrector_step(
        vn=vn,
        w=w,
        tangential_wind=tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        coriolis_frequency=coriolis_frequency,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        geofac_grdiv=geofac_grdiv,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=(
            vertical_wind_advective_tendency,
            vertical_cfl,
            normal_wind_advective_tendency,
        ),
        domain=(
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )
