# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.atmosphere.dycore.dycore_states import (
    DivergenceDampingOrder,
    HorizontalPressureDiscretizationType,
    RhoThetaAdvectionType,
)
from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    _add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    _add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    _add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    _apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    _apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.apply_weighted_2nd_and_4th_order_divergence_damping import (
    _apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import _compute_graddiv2_of_vn
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_gradients_grad_green_gauss_cell_dsl import (
    _mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


rhotheta_avd_type: Final = RhoThetaAdvectionType()
horzpres_discr_type: Final = HorizontalPressureDiscretizationType()
divergence_damp_order: Final = DivergenceDampingOrder()


@gtx.field_operator
def _compute_theta_rho_face_values_and_pressure_gradient_and_update_vn_in_predictor_step(
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    perturbed_rho: fa.CellKField[ta.vpfloat],
    perturbed_theta_v: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    hydrostatic_correction_on_lowest_level: fa.EdgeField[ta.wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increments: fa.EdgeKField[ta.vpfloat],
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    pos_on_tplane_e_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    pos_on_tplane_e_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    dual_normal_cell_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    dual_normal_cell_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ikoffset: gtx.Field[[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.ECDim, dims.KDim], ta.vpfloat],
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    dtime: ta.wpfloat,
    cpd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    is_iau_active: bool,
    limited_area: bool,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_edge_halo_level_2: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    end_edge_halo: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_end: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
]:
    (
        ddx_perturbed_rho,
        ddy_perturbed_rho,
        ddx_perturbed_theta_v,
        ddy_perturbed_theta_v,
    ) = (
        _mo_math_gradients_grad_green_gauss_cell_dsl(
            p_ccpr1=perturbed_rho,
            p_ccpr2=perturbed_theta_v,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
        )
        if (iadv_rhotheta == rhotheta_avd_type.MIURA)
        else (
            broadcast(0.0, (dims.CellDim, dims.KDim)),
            broadcast(0.0, (dims.CellDim, dims.KDim)),
            broadcast(0.0, (dims.CellDim, dims.KDim)),
            broadcast(0.0, (dims.CellDim, dims.KDim)),
        )
    )

    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = (
        concat_where(
            (start_edge_halo_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_halo_level_2),
            (
                broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
            ),
            (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
        )
        if iadv_rhotheta <= 2
        else (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels)
    )

    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = (
        concat_where(
            (start_edge_lateral_boundary <= dims.EdgeDim) & (dims.EdgeDim < end_edge_halo),
            (
                broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
            ),
            (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
        )
        if limited_area & (iadv_rhotheta <= 2)
        else (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels)
    )

    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = (
        concat_where(
            (start_edge_lateral_boundary_level_7 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_halo),
            _compute_horizontal_advection_of_rho_and_theta(
                p_vn=current_vn,
                p_vt=tangential_wind,
                pos_on_tplane_e_1=pos_on_tplane_e_x,
                pos_on_tplane_e_2=pos_on_tplane_e_y,
                primal_normal_cell_1=primal_normal_cell_x,
                dual_normal_cell_1=dual_normal_cell_x,
                primal_normal_cell_2=primal_normal_cell_y,
                dual_normal_cell_2=dual_normal_cell_y,
                p_dthalf=wpfloat("0.5") * dtime,
                rho_ref_me=reference_rho_at_edges_on_model_levels,
                theta_ref_me=reference_theta_at_edges_on_model_levels,
                z_grad_rth_1=ddx_perturbed_rho,
                z_grad_rth_2=ddy_perturbed_rho,
                z_grad_rth_3=ddx_perturbed_theta_v,
                z_grad_rth_4=ddy_perturbed_theta_v,
                z_rth_pr_1=perturbed_rho,
                z_rth_pr_2=perturbed_theta_v,
            ),
            (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels),
        )
        if (iadv_rhotheta == rhotheta_avd_type.MIURA) & (iadv_rhotheta <= 2)
        else (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels)
    )

    horizontal_pressure_gradient = concat_where(
        dims.KDim < nflatlev,
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
                inv_dual_edge_length=inv_dual_edge_length,
                z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
            ),
            horizontal_pressure_gradient,
        ),
        horizontal_pressure_gradient,
    )

    horizontal_pressure_gradient = (
        concat_where(
            (nflatlev <= dims.KDim) & (dims.KDim < (nflat_gradp + 1)),
            concat_where(
                (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
                _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
                    ddxn_z_full=ddxn_z_full,
                    c_lin_e=c_lin_e,
                    z_dexner_dz_c_1=ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                ),
                horizontal_pressure_gradient,
            ),
            horizontal_pressure_gradient,
        )
        if (igradp_method == horzpres_discr_type.TAYLOR_HYDRO) & (nflatlev < (nflat_gradp + 1))
        else horizontal_pressure_gradient
    )

    horizontal_pressure_gradient = (
        concat_where(
            (nflat_gradp + 1) <= dims.KDim,
            concat_where(
                (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
                _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
                    inv_dual_edge_length=inv_dual_edge_length,
                    z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
                    zdiff_gradp=zdiff_gradp,
                    ikoffset=ikoffset,
                    z_dexner_dz_c_1=ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                    z_dexner_dz_c_2=d2dz2_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                ),
                horizontal_pressure_gradient,
            ),
            horizontal_pressure_gradient,
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else horizontal_pressure_gradient
    )

    horizontal_pressure_gradient = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_end),
            _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
                ipeidx_dsl=ipeidx_dsl,
                pg_exdist=pg_exdist,
                z_hydro_corr=hydrostatic_correction_on_lowest_level,
                z_gradh_exner=horizontal_pressure_gradient,
            ),
            horizontal_pressure_gradient,
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else horizontal_pressure_gradient
    )

    next_vn = concat_where(
        (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
        _add_temporal_tendencies_to_vn(
            vn_nnow=current_vn,
            ddt_vn_apc_ntl1=predictor_normal_wind_advective_tendency,
            ddt_vn_phy=normal_wind_tendency_due_to_physics_process,
            z_theta_v_e=theta_v_at_edges_on_model_levels,
            z_gradh_exner=horizontal_pressure_gradient,
            dtime=dtime,
            cpd=cpd,
        ),
        next_vn,
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _add_analysis_increments_to_vn(
                vn_incr=normal_wind_iau_increments, vn=next_vn, iau_wgt_dyn=iau_wgt_dyn
            ),
            next_vn,
        )
        if is_iau_active
        else next_vn
    )

    return (
        rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient,
        next_vn,
    )


@gtx.field_operator
def _apply_divergence_damping_and_update_vn_in_corrector_step(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    corrector_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increments: fa.EdgeKField[ta.vpfloat],
    reduced_fourth_order_divdamp_coeff_at_nest_boundary: fa.KField[ta.wpfloat],
    fourth_order_divdamp_scaling_coeff: fa.KField[ta.wpfloat],
    second_order_divdamp_scaling_coeff: ta.wpfloat,
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    nudgecoeff_e: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    fourth_order_divdamp_factor: ta.wpfloat,
    second_order_divdamp_factor: ta.wpfloat,
    wgt_nnow_vel: ta.wpfloat,
    wgt_nnew_vel: ta.wpfloat,
    dtime: ta.wpfloat,
    cpd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    is_iau_active: bool,
    itime_scheme: gtx.int32,
    limited_area: bool,
    divdamp_order: gtx.int32,
    starting_index_for_3d_divdamp: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
) -> fa.EdgeKField[ta.wpfloat]:
    horizontal_gradient_of_total_divergence = concat_where(
        starting_index_for_3d_divdamp <= dims.KDim,
        concat_where(
            (start_edge_lateral_boundary_level_7 <= dims.EdgeDim)
            & (dims.EdgeDim < end_edge_halo_level_2),
            _add_vertical_wind_derivative_to_divergence_damping(
                hmask_dd3d=horizontal_mask_for_3d_divdamp,
                scalfac_dd3d=scaling_factor_for_3d_divdamp,
                inv_dual_edge_length=inv_dual_edge_length,
                z_dwdz_dd=dwdz_at_cells_on_model_levels,
                z_graddiv_vn=horizontal_gradient_of_normal_wind_divergence,
            ),
            horizontal_gradient_of_normal_wind_divergence,
        ),
        horizontal_gradient_of_normal_wind_divergence,
    )

    squared_horizontal_gradient_of_total_divergence = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _compute_graddiv2_of_vn(
                geofac_grdiv=geofac_grdiv, z_graddiv_vn=horizontal_gradient_of_total_divergence
            ),
            broadcast(vpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
        )
        if (divdamp_order == divergence_damp_order.COMBINED)
        | (divdamp_order == divergence_damp_order.FOURTH_ORDER)
        else broadcast(vpfloat("0.0"), (dims.EdgeDim, dims.KDim))
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
                vn_nnow=current_vn,
                ddt_vn_apc_ntl1=predictor_normal_wind_advective_tendency,
                ddt_vn_apc_ntl2=corrector_normal_wind_advective_tendency,
                ddt_vn_phy=normal_wind_tendency_due_to_physics_process,
                z_theta_v_e=theta_v_at_edges_on_model_levels,
                z_gradh_exner=horizontal_pressure_gradient,
                dtime=dtime,
                wgt_nnow_vel=wgt_nnow_vel,
                wgt_nnew_vel=wgt_nnew_vel,
                cpd=cpd,
            ),
            next_vn,
        )
        if itime_scheme == 4
        else next_vn
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _apply_2nd_order_divergence_damping(
                z_graddiv_vn=horizontal_gradient_of_total_divergence,
                vn=next_vn,
                scal_divdamp_o2=second_order_divdamp_scaling_coeff,
            ),
            next_vn,
        )
        if (
            (divdamp_order == divergence_damp_order.COMBINED)
            & (second_order_divdamp_scaling_coeff > 1.0e-6)
        )
        else next_vn
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _apply_weighted_2nd_and_4th_order_divergence_damping(
                scal_divdamp=fourth_order_divdamp_scaling_coeff,
                bdy_divdamp=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
                nudgecoeff_e=nudgecoeff_e,
                z_graddiv2_vn=squared_horizontal_gradient_of_total_divergence,
                vn=next_vn,
            ),
            next_vn,
        )
        if (
            (divdamp_order == divergence_damp_order.COMBINED)
            & (second_order_divdamp_factor <= (4.0 * fourth_order_divdamp_factor))
            & limited_area
        )
        else next_vn
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _apply_4th_order_divergence_damping(
                scal_divdamp=fourth_order_divdamp_scaling_coeff,
                z_graddiv2_vn=squared_horizontal_gradient_of_total_divergence,
                vn=next_vn,
            ),
            next_vn,
        )
        if (
            (divdamp_order == divergence_damp_order.COMBINED)
            & (second_order_divdamp_factor <= (4.0 * fourth_order_divdamp_factor))
            & (not limited_area)
        )
        else next_vn
    )

    next_vn = (
        concat_where(
            (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
            _add_analysis_increments_to_vn(
                vn_incr=normal_wind_iau_increments, vn=next_vn, iau_wgt_dyn=iau_wgt_dyn
            ),
            next_vn,
        )
        if is_iau_active
        else next_vn
    )

    return next_vn


@gtx.program(grid_type=GridType.UNSTRUCTURED)
def compute_theta_rho_face_values_and_pressure_gradient_and_update_vn_in_predictor_step(
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    perturbed_rho: fa.CellKField[ta.vpfloat],
    perturbed_theta_v: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    hydrostatic_correction_on_lowest_level: fa.EdgeField[ta.wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increments: fa.EdgeKField[ta.vpfloat],
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    pos_on_tplane_e_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    pos_on_tplane_e_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    dual_normal_cell_x: gtx.Field[[dims.ECDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    dual_normal_cell_y: gtx.Field[[dims.ECDim], ta.wpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ikoffset: gtx.Field[[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.ECDim, dims.KDim], ta.vpfloat],
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    dtime: ta.wpfloat,
    cpd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    is_iau_active: bool,
    limited_area: bool,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_edge_halo_level_2: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    end_edge_halo: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_end: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """Formerly known as fused_solve_nonhydro_stencil_15_to_28_predictor."""
    _compute_theta_rho_face_values_and_pressure_gradient_and_update_vn_in_predictor_step(
        rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        next_vn=next_vn,
        current_vn=current_vn,
        tangential_wind=tangential_wind,
        reference_rho_at_edges_on_model_levels=reference_rho_at_edges_on_model_levels,
        reference_theta_at_edges_on_model_levels=reference_theta_at_edges_on_model_levels,
        perturbed_rho=perturbed_rho,
        perturbed_theta_v=perturbed_theta_v,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_physics_process=normal_wind_tendency_due_to_physics_process,
        normal_wind_iau_increments=normal_wind_iau_increments,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        pos_on_tplane_e_x=pos_on_tplane_e_x,
        pos_on_tplane_e_y=pos_on_tplane_e_y,
        primal_normal_cell_x=primal_normal_cell_x,
        dual_normal_cell_x=dual_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_y=dual_normal_cell_y,
        ddxn_z_full=ddxn_z_full,
        c_lin_e=c_lin_e,
        ikoffset=ikoffset,
        zdiff_gradp=zdiff_gradp,
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        inv_dual_edge_length=inv_dual_edge_length,
        dtime=dtime,
        cpd=cpd,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        limited_area=limited_area,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        start_edge_halo_level_2=start_edge_halo_level_2,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        end_edge_halo=end_edge_halo,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        end_edge_end=end_edge_end,
        out=(
            rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient,
            next_vn,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=GridType.UNSTRUCTURED)
def apply_divergence_damping_and_update_vn_in_corrector_step(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    corrector_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increments: fa.EdgeKField[ta.vpfloat],
    reduced_fourth_order_divdamp_coeff_at_nest_boundary: fa.KField[ta.wpfloat],
    fourth_order_divdamp_scaling_coeff: fa.KField[ta.wpfloat],
    second_order_divdamp_scaling_coeff: ta.wpfloat,
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    nudgecoeff_e: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    fourth_order_divdamp_factor: ta.wpfloat,
    second_order_divdamp_factor: ta.wpfloat,
    wgt_nnow_vel: ta.wpfloat,
    wgt_nnew_vel: ta.wpfloat,
    dtime: ta.wpfloat,
    cpd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    is_iau_active: bool,
    itime_scheme: gtx.int32,
    limited_area: bool,
    divdamp_order: gtx.int32,
    starting_index_for_3d_divdamp: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """Formerly known as fused_solve_nonhydro_stencil_15_to_28_corrector."""
    _apply_divergence_damping_and_update_vn_in_corrector_step(
        horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
        next_vn=next_vn,
        current_vn=current_vn,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_physics_process=normal_wind_tendency_due_to_physics_process,
        normal_wind_iau_increments=normal_wind_iau_increments,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        reduced_fourth_order_divdamp_coeff_at_nest_boundary=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        fourth_order_divdamp_scaling_coeff=fourth_order_divdamp_scaling_coeff,
        second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
        horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
        scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
        inv_dual_edge_length=inv_dual_edge_length,
        nudgecoeff_e=nudgecoeff_e,
        geofac_grdiv=geofac_grdiv,
        fourth_order_divdamp_factor=fourth_order_divdamp_factor,
        second_order_divdamp_factor=second_order_divdamp_factor,
        wgt_nnow_vel=wgt_nnow_vel,
        wgt_nnew_vel=wgt_nnew_vel,
        dtime=dtime,
        cpd=cpd,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        itime_scheme=itime_scheme,
        limited_area=limited_area,
        divdamp_order=divdamp_order,
        starting_index_for_3d_divdamp=starting_index_for_3d_divdamp,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        out=next_vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
