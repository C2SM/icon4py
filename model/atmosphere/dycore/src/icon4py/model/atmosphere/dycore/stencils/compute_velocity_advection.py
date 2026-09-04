# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_horizontal_momentum_equation import (
    _compute_advection_in_horizontal_momentum,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    _compute_advection_in_corrector_vertical_momentum,
    _compute_advection_in_predictor_vertical_momentum,
)
from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    _compute_diagnostics_from_normal_wind,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


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
                dims.KHalfDim: (vertical_start, vertical_end),
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
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
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
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
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
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
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
                dims.KHalfDim: (vertical_start, vertical_end),
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
