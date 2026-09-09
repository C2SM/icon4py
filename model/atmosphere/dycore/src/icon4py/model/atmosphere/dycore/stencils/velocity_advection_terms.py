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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E, C2E2CO, E2C, E2C2EO, E2V
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center_vp import (
    _interpolate_to_cell_center_vp,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


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
def _compute_vertical_advection_of_w(
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
def _compute_interpolated_horizontal_advection_of_w(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_17."""
    horizontal_advection_of_w_at_edges_on_half_levels_wp = astype(
        horizontal_advection_of_w_at_edges_on_half_levels, wpfloat
    )
    return neighbor_sum(
        horizontal_advection_of_w_at_edges_on_half_levels_wp(C2E) * e_bln_c_s, axis=dims.C2EDim
    )


@gtx.field_operator
def _compute_extra_diffusion_for_w(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_18."""
    contravariant_corrected_w_at_cells_on_half_levels_wp, ddqz_z_half_wp, cfl_w_limit_wp = astype(
        (contravariant_corrected_w_at_cells_on_half_levels, ddqz_z_half, cfl_w_limit), wpfloat
    )

    difcoef = scalfac_exdiff * minimum(
        wpfloat("0.85") - cfl_w_limit_wp * dtime,
        abs(contravariant_corrected_w_at_cells_on_half_levels_wp) * dtime / ddqz_z_half_wp
        - cfl_w_limit_wp * dtime,
    )

    return difcoef * area * neighbor_sum(w(C2E2CO) * geofac_n2s, axis=dims.C2E2CODim)


@gtx.field_operator
def _compute_cfl(
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> tuple[fa.CellKHalfField[bool], fa.CellKHalfField[ta.vpfloat]]:
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

    return cfl_clipping, astype(vertical_cfl, vpfloat)


@gtx.field_operator
def _clip_contravariant_corrected_w(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    cfl_clipping: fa.CellKHalfField[bool],
    vertical_cfl: fa.CellKHalfField[ta.vpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels_wp = astype(
        contravariant_corrected_w_at_cells_on_half_levels, wpfloat
    )

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl < -vpfloat("0.85")),
        astype(-vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl > vpfloat("0.85")),
        astype(vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    return astype(contravariant_corrected_w_at_cells_on_half_levels_wp, vpfloat)


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

    cfl_clipping, vertical_cfl = concat_where(
        (dims.KHalfDim >= maximum(2, end_index_of_damping_layer - 2)) & (dims.KHalfDim < nlev - 3),
        _compute_cfl(
            ddqz_z_half=ddqz_z_half,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        ),
        (
            broadcast(False, (dims.CellDim, dims.KHalfDim)),
            broadcast(vpfloat("0.0"), (dims.CellDim, dims.KHalfDim)),
        ),
    )

    contravariant_corrected_w_at_cells_on_half_levels = _clip_contravariant_corrected_w(
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
        ddqz_z_half,
        dtime,
    )

    return contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl


@gtx.field_operator
def _compute_advective_vertical_wind_tendency(
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
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
    # TODO(havogt): the wp-vp roundtrips are here to be faithful to ICON's mixed precision,
    # but was that a deliberate decision on the Fortran side?
    vertical_advection_of_w = _compute_vertical_advection_of_w(
        contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
    )
    horizontal_advection_of_w_at_edges_on_half_levels = _compute_horizontal_advection_of_w(
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
    )
    interpolated_horizontal_advection_of_w = _compute_interpolated_horizontal_advection_of_w(
        e_bln_c_s, horizontal_advection_of_w_at_edges_on_half_levels
    )
    extra_diffusion_for_w = _compute_extra_diffusion_for_w(
        contravariant_corrected_w_at_cells_on_half_levels,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
    )

    vertical_wind_advective_tendency = astype(
        astype(vertical_advection_of_w, wpfloat) + interpolated_horizontal_advection_of_w,
        vpfloat,
    )

    vertical_wind_advective_tendency_wp = astype(vertical_wind_advective_tendency, wpfloat)
    vertical_wind_advective_tendency = astype(
        where(
            cfl_clipping & owner_mask,
            vertical_wind_advective_tendency_wp + extra_diffusion_for_w,
            vertical_wind_advective_tendency_wp,
        ),
        vpfloat,
    )

    return vertical_wind_advective_tendency


@gtx.field_operator
def _compute_advection_in_vertical_momentum(
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
    skip_vertical_wind_advective_tendency: bool,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKHalfField[ta.vpfloat], fa.CellKField[ta.vpfloat], fa.CellKHalfField[ta.vpfloat]]:
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

    vertical_wind_advective_tendency = (
        _compute_advective_vertical_wind_tendency(
            w=w,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
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
        if not skip_vertical_wind_advective_tendency
        else broadcast(
            vpfloat("0.0"),
            (dims.CellDim, dims.KHalfDim),
        )  # "0.0" is a dummy value and should never be used
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
