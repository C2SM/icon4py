# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype, broadcast, int32

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    _add_analysis_increments_from_data_assimilation,
)
from icon4py.model.atmosphere.dycore.stencils.apply_rayleigh_damping_mechanism import (
    _apply_rayleigh_damping_mechanism,
)
from icon4py.model.atmosphere.dycore.stencils.compute_divergence_of_fluxes_of_rho_and_theta import (
    _compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    _compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_explicit_part_for_rho_and_exner import (
    _compute_explicit_part_for_rho_and_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_solver_coefficients_matrix import (
    _compute_solver_coefficients_matrix,
)
from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    _compute_results_for_thermodynamic_variables,
)
from icon4py.model.atmosphere.dycore.stencils.set_lower_boundary_condition_for_w_and_contravariant_correction import (
    _set_lower_boundary_condition_for_w_and_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_back_substitution import (
    _solve_tridiagonal_matrix_for_w_back_substitution_scan,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    _solve_tridiagonal_matrix_for_w_forward_sweep_2,
)
from icon4py.model.atmosphere.dycore.stencils.update_dynamical_exner_time_increment import (
    _update_dynamical_exner_time_increment,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_volume_flux import (
    _update_mass_volume_flux,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _set_surface_boundary_condtion_for_computation_of_w(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    tridiagonal_alpha_coeff_at_cells_on_half_levels = broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))
    (w, vertical_mass_flux_at_cells_on_half_levels) = _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c=contravariant_correction_at_cells_on_half_levels
    )

    return tridiagonal_alpha_coeff_at_cells_on_half_levels, w, vertical_mass_flux_at_cells_on_half_levels


@gtx.field_operator
def _compute_w_explicit_term_with_predictor_advective_tendency(
    current_w: fa.CellKField[wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> fa.CellKField[wpfloat]:
    predictor_vertical_wind_advective_tendency_wp, z_th_ddz_exner_c_wp = astype(
        (predictor_vertical_wind_advective_tendency, z_th_ddz_exner_c), wpfloat
    )

    w_explicit_term_wp = current_w + dtime * (predictor_vertical_wind_advective_tendency_wp - cpd * z_th_ddz_exner_c_wp)
    return w_explicit_term_wp


@gtx.field_operator
def _compute_w_explicit_term_with_interpolated_predictor_corrector_advective_tendency(
    current_w: fa.CellKField[wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    dtime: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    cpd: wpfloat,
) -> fa.CellKField[wpfloat]:
    predictor_vertical_wind_advective_tendency_wp, corrector_vertical_wind_advective_tendency_wp, z_th_ddz_exner_c_wp = astype(
        (predictor_vertical_wind_advective_tendency, corrector_vertical_wind_advective_tendency, z_th_ddz_exner_c), wpfloat
    )

    w_explicit_term_wp = current_w + dtime * (
        wgt_nnow_vel * predictor_vertical_wind_advective_tendency_wp
        + wgt_nnew_vel * corrector_vertical_wind_advective_tendency_wp
        - cpd * z_th_ddz_exner_c_wp
    )
    return w_explicit_term_wp


@gtx.field_operator
def _vertically_implicit_solver_at_predictor_step_before_solving_w(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.vpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    ddt_exner_phy: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
    cpd: ta.wpfloat,
    l_vert_nested: bool,
    is_iau_active: bool,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    divergence_of_mass, divergence_of_theta_v = _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div=geofac_div,
        mass_fl_e=mass_flux_at_edges_on_model_levels,
        z_theta_v_fl_e=theta_v_flux_at_edges_on_model_levels,
    )

    tridiagonal_intermediate_result = broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))

    w_explicit_term = concat_where(
        1 <= dims.KDim,
        _compute_w_explicit_term_with_predictor_advective_tendency(
            current_w=current_w,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            dtime=dtime,
            cpd=cpd,
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    )

    vertical_mass_flux_at_cells_on_half_levels = concat_where(
        1 <= dims.KDim,
        rho_ic * (-astype(contravariant_correction_at_cells_on_half_levels, wpfloat) + vwind_expl_wgt * current_w),
        vertical_mass_flux_at_cells_on_half_levels
    )

    (tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels) = _compute_solver_coefficients_matrix(
        current_exner,
        current_rho,
        current_theta_v,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_at_cells_on_half_levels,
        rho_ic,
        dtime,
        rd,
        cvd,
    )

    (next_w, vertical_mass_flux_at_cells_on_half_levels) = (
        concat_where(
            dims.KDim == 0,
            (
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            ),
            (next_w, vertical_mass_flux_at_cells_on_half_levels),
        )
        if (not l_vert_nested)
        else (next_w, vertical_mass_flux_at_cells_on_half_levels)
    )

    (rho_explicit_term, exner_explicit_term) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
        exner_pr=exner_pr,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
    )

    rho_explicit_term, exner_explicit_term = (
        _add_analysis_increments_from_data_assimilation(
            z_rho_expl=rho_explicit_term,
            z_exner_expl=exner_explicit_term,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        if is_iau_active
        else (rho_explicit_term, exner_explicit_term)
    )

    tridiagonal_intermediate_result, next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_forward_sweep_2(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
            z_w_expl=w_explicit_term,
            z_exner_expl=exner_explicit_term,
            z_q=tridiagonal_intermediate_result,
            w=next_w,
            dtime=dtime,
            cpd=cpd,
        ),
        (tridiagonal_intermediate_result, next_w),
    )

    next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(z_q=tridiagonal_intermediate_result, w=next_w),
        next_w,
    )

    return (
        vertical_mass_flux_at_cells_on_half_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        next_w,
        rho_explicit_term,
        exner_explicit_term,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_predictor_step_after_solving_w(
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    exner_dynamical_increment: fa.CellKField[ta.vpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    z_raylfac: fa.KField[ta.wpfloat],
    exner_ref_mc: fa.CellKField[ta.vpfloat],
    rho_explicit_term: fa.CellKField[ta.wpfloat],
    exner_explicit_term: fa.CellKField[ta.wpfloat],
    cvd_o_rd: ta.wpfloat,
    dtime: ta.wpfloat,
    rayleigh_klemp: int32,
    rayleigh_type: int32,
    divdamp_type: int32,
    at_first_substep: bool,
    index_of_damping_layer: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # Because we do not support nesting, it is safe to assume w_1 is a zero field
    w_1 = broadcast(wpfloat("0.0"), (dims.CellDim,))
    next_w = (
        concat_where(
            (dims.KDim > 0) & (dims.KDim < index_of_damping_layer + 1),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=next_w,
            ),
            next_w,
        )
        if rayleigh_type == rayleigh_klemp
        else next_w
    )

    next_rho, next_exner, next_theta_v = concat_where(
        jk_start <= dims.KDim,
        _compute_results_for_thermodynamic_variables(
            z_rho_expl=rho_explicit_term,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=next_w,
            z_exner_expl=exner_explicit_term,
            exner_ref_mc=exner_ref_mc,
            z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
            rho_now=current_rho,
            theta_v_now=current_theta_v,
            exner_now=current_exner,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        ),
        (next_rho, next_exner, next_theta_v),
    )

    # compute dw/dz for divergence damping term
    dwdz_at_cells_on_model_levels = (
        concat_where(
            kstart_dd3d <= dims.KDim,
            _compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=inv_ddqz_z_full,
                w=next_w,
                w_concorr_c=contravariant_correction_at_cells_on_half_levels,
            ),
            dwdz_at_cells_on_model_levels,
        )
        if divdamp_type >= 3
        else dwdz_at_cells_on_model_levels
    )

    exner_dynamical_increment = (
        concat_where(
            kstart_moist <= dims.KDim,
            astype(current_exner, vpfloat),
            exner_dynamical_increment,
        )
        if at_first_substep
        else exner_dynamical_increment
    )

    return (
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        next_w,
        rho_explicit_term,
        exner_explicit_term,
        next_rho,
        next_exner,
        next_theta_v,
        dwdz_at_cells_on_model_levels,
        exner_dynamical_increment,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_corrector_step_before_solving_w(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.vpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    ddt_exner_phy: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    wgt_nnow_vel: ta.wpfloat,
    wgt_nnew_vel: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
    cpd: ta.wpfloat,
    is_iau_active: bool,
    l_vert_nested: bool,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # verified for e-9
    divergence_of_mass, divergence_of_theta_v = _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div=geofac_div,
        mass_fl_e=mass_flux_at_edges_on_model_levels,
        z_theta_v_fl_e=theta_v_flux_at_edges_on_model_levels,
    )

    tridiagonal_intermediate_result = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))

    w_explicit_term = concat_where(
        1 <= dims.KDim,
        _compute_w_explicit_term_with_interpolated_predictor_corrector_advective_tendency(
            current_w=current_w,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    )

    vertical_mass_flux_at_cells_on_half_levels = concat_where(
        1 <= dims.KDim,
        rho_ic * (-astype(contravariant_correction_at_cells_on_half_levels, wpfloat) + vwind_expl_wgt * current_w),
        vertical_mass_flux_at_cells_on_half_levels
    )

    (tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels) = _compute_solver_coefficients_matrix(
        current_exner,
        current_rho,
        current_theta_v,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_at_cells_on_half_levels,
        rho_ic,
        dtime,
        rd,
        cvd,
    )

    (next_w, vertical_mass_flux_at_cells_on_half_levels) = (
        concat_where(
            dims.KDim == 0,
            (
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            ),
            (next_w, vertical_mass_flux_at_cells_on_half_levels),
        )
        if (not l_vert_nested)
        else (next_w, vertical_mass_flux_at_cells_on_half_levels)
    )

    (rho_explicit_term, exner_explicit_term) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
        exner_pr=exner_pr,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
    )

    rho_explicit_term, exner_explicit_term = (
        _add_analysis_increments_from_data_assimilation(
            z_rho_expl=rho_explicit_term,
            z_exner_expl=exner_explicit_term,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        if is_iau_active
        else (rho_explicit_term, exner_explicit_term)
    )

    tridiagonal_intermediate_result, next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_forward_sweep_2(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
            z_w_expl=w_explicit_term,
            z_exner_expl=exner_explicit_term,
            z_q=tridiagonal_intermediate_result,
            w=next_w,
            dtime=dtime,
            cpd=cpd,
        ),
        (tridiagonal_intermediate_result, next_w),
    )

    next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(
            z_q=tridiagonal_intermediate_result,
            w=next_w,
        ),
        next_w,
    )

    return (
        vertical_mass_flux_at_cells_on_half_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        next_w,
        rho_explicit_term,
        exner_explicit_term,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_corrector_step_after_solving_w(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    ddt_exner_phy: fa.CellKField[ta.vpfloat],
    z_raylfac: fa.KField[ta.wpfloat],
    exner_ref_mc: fa.CellKField[ta.vpfloat],
    rho_explicit_term: fa.CellKField[ta.wpfloat],
    exner_explicit_term: fa.CellKField[ta.wpfloat],
    lprep_adv: bool,
    r_nsubsteps: ta.wpfloat,
    ndyn_substeps_var: ta.wpfloat,
    cvd_o_rd: ta.wpfloat,
    dtime: ta.wpfloat,
    rayleigh_klemp: int32,
    rayleigh_type: int32,
    index_of_damping_layer: int32,
    jk_start: int32,
    kstart_moist: int32,
    at_first_substep: bool,
    at_last_substep: bool,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    # Because we do not support nesting, it is safe to assume w_1 is a zero field
    w_1 = broadcast(wpfloat("0.0"), (dims.CellDim,))
    next_w = (
        concat_where(
            (dims.KDim > 0) & (dims.KDim < index_of_damping_layer + 1),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=next_w,
            ),
            next_w,
        )
        if rayleigh_type == rayleigh_klemp
        else next_w
    )

    next_rho, next_exner, next_theta_v = concat_where(
        jk_start <= dims.KDim,
        _compute_results_for_thermodynamic_variables(
            z_rho_expl=rho_explicit_term,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=next_w,
            z_exner_expl=exner_explicit_term,
            exner_ref_mc=exner_ref_mc,
            z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
            rho_now=current_rho,
            theta_v_now=current_theta_v,
            exner_now=current_exner,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        ),
        (next_rho, next_exner, next_theta_v),
    )

    dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels = (
        (
            broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
        )
        if (lprep_adv & at_first_substep)
        else (dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels)
    )

    dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels = concat_where(
        1 <= dims.KDim,
        _update_mass_volume_flux(
            z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=next_w,
            mass_flx_ic=dynamical_vertical_mass_flux_at_cells_on_half_levels,
            vol_flx_ic=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            r_nsubsteps=r_nsubsteps,
        ),
        (dynamical_vertical_mass_flux_at_cells_on_half_levels, dynamical_vertical_volumetric_flux_at_cells_on_half_levels),
    )

    exner_dynamical_increment = (
        concat_where(
            dims.KDim >= kstart_moist,
            _update_dynamical_exner_time_increment(
                exner=next_exner,
                ddt_exner_phy=ddt_exner_phy,
                exner_dyn_incr=exner_dynamical_increment,
                ndyn_substeps_var=ndyn_substeps_var,
                dtime=dtime,
            ),
            exner_dynamical_increment,
        )
        if at_last_substep
        else exner_dynamical_increment
    )

    return (
        next_w,
        next_rho,
        next_exner,
        next_theta_v,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        exner_dynamical_increment,
    )


@gtx.program
def vertically_implicit_solver_at_predictor_step(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    rho_explicit_term: fa.CellKField[ta.wpfloat],
    exner_explicit_term: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    exner_dynamical_increment: fa.CellKField[ta.vpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.vpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    ddt_exner_phy: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    z_raylfac: fa.KField[ta.wpfloat],
    exner_ref_mc: fa.CellKField[ta.vpfloat],
    cvd_o_rd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
    cpd: ta.wpfloat,
    rayleigh_klemp: int32,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    divdamp_type: int32,
    at_first_substep: bool,
    index_of_damping_layer: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_surface_boundary_condtion_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=(
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w,
            vertical_mass_flux_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )

    _vertically_implicit_solver_at_predictor_step_before_solving_w(
        vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
        next_w=next_w,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        vwind_expl_wgt=vwind_expl_wgt,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        exner_pr=exner_pr,
        ddt_exner_phy=ddt_exner_phy,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=ddqz_z_half,
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=dtime,
        rd=rd,
        cvd=cvd,
        cpd=cpd,
        l_vert_nested=l_vert_nested,
        is_iau_active=is_iau_active,
        out=(
            vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w,
            rho_explicit_term,
            exner_explicit_term,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _vertically_implicit_solver_at_predictor_step_after_solving_w(
        tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        next_w=next_w,
        next_rho=next_rho,
        next_exner=next_exner,
        next_theta_v=next_theta_v,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        rho_ic=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        inv_ddqz_z_full=inv_ddqz_z_full,
        vwind_impl_wgt=vwind_impl_wgt,
        z_raylfac=z_raylfac,
        exner_ref_mc=exner_ref_mc,
        rho_explicit_term=rho_explicit_term,
        exner_explicit_term=exner_explicit_term,
        cvd_o_rd=cvd_o_rd,
        dtime=dtime,
        rayleigh_klemp=rayleigh_klemp,
        rayleigh_type=rayleigh_type,
        divdamp_type=divdamp_type,
        at_first_substep=at_first_substep,
        index_of_damping_layer=index_of_damping_layer,
        jk_start=jk_start,
        kstart_dd3d=kstart_dd3d,
        kstart_moist=kstart_moist,
        out=(
            tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w,
            rho_explicit_term,
            exner_explicit_term,
            next_rho,
            next_exner,
            next_theta_v,
            dwdz_at_cells_on_model_levels,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )


@gtx.program
def vertically_implicit_solver_at_corrector_step(
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    rho_explicit_term: fa.CellKField[ta.wpfloat],
    exner_explicit_term: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    z_th_ddz_exner_c: fa.CellKField[ta.vpfloat],
    rho_ic: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    vwind_impl_wgt: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_pr: fa.CellKField[ta.wpfloat],
    ddt_exner_phy: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    z_raylfac: fa.KField[ta.wpfloat],
    exner_ref_mc: fa.CellKField[ta.vpfloat],
    wgt_nnow_vel: ta.wpfloat,
    wgt_nnew_vel: ta.wpfloat,
    lprep_adv: bool,
    r_nsubsteps: ta.wpfloat,
    ndyn_substeps_var: ta.wpfloat,
    cvd_o_rd: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
    cpd: ta.wpfloat,
    rayleigh_klemp: int32,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    at_first_substep: bool,
    at_last_substep: bool,
    index_of_damping_layer: int32,
    jk_start: int32,
    kstart_moist: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_surface_boundary_condtion_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=(
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w,
            vertical_mass_flux_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
    _vertically_implicit_solver_at_corrector_step_before_solving_w(
        vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
        next_w=next_w,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        rho_ic=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        vwind_expl_wgt=vwind_expl_wgt,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        exner_pr=exner_pr,
        ddt_exner_phy=ddt_exner_phy,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=ddqz_z_half,
        wgt_nnow_vel=wgt_nnow_vel,
        wgt_nnew_vel=wgt_nnew_vel,
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=dtime,
        rd=rd,
        cvd=cvd,
        cpd=cpd,
        is_iau_active=is_iau_active,
        l_vert_nested=l_vert_nested,
        out=(
            vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w,
            rho_explicit_term,
            exner_explicit_term,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _vertically_implicit_solver_at_corrector_step_after_solving_w(
        vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        next_w=next_w,
        next_rho=next_rho,
        next_exner=next_exner,
        next_theta_v=next_theta_v,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        rho_ic=rho_ic,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        inv_ddqz_z_full=inv_ddqz_z_full,
        vwind_impl_wgt=vwind_impl_wgt,
        ddt_exner_phy=ddt_exner_phy,
        z_raylfac=z_raylfac,
        exner_ref_mc=exner_ref_mc,
        rho_explicit_term=rho_explicit_term,
        exner_explicit_term=exner_explicit_term,
        lprep_adv=lprep_adv,
        r_nsubsteps=r_nsubsteps,
        ndyn_substeps_var=ndyn_substeps_var,
        cvd_o_rd=cvd_o_rd,
        dtime=dtime,
        rayleigh_klemp=rayleigh_klemp,
        rayleigh_type=rayleigh_type,
        index_of_damping_layer=index_of_damping_layer,
        jk_start=jk_start,
        kstart_moist=kstart_moist,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        out=(
            next_w,
            next_rho,
            next_exner,
            next_theta_v,
            dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
