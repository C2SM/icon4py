# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype, broadcast, int32

from icon4py.model.atmosphere.dycore.solve_nonhydro_stencils import (
    _stencils_42_44_45,
    _stencils_43_44_45,
)
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


@gtx.scan_operator(axis=dims.KDim, forward=True, init=0.0)
def _w_1_scan(state: ta.wpfloat, w_1: ta.wpfloat) -> ta.wpfloat:
    return w_1 + state


@gtx.field_operator
def _set_surface_boundary_condtion_for_computation_of_w(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    z_alpha = broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim))
    (w, z_contr_w_fl_l) = _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c=contravariant_correction_at_cells_on_half_levels
    )

    return z_alpha, w, z_contr_w_fl_l


@gtx.field_operator
def _vertically_implicit_solver_at_predictor_step_before_solving_w(
    z_contr_w_fl_l: fa.CellKField[ta.wpfloat],
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    geofac_div: fa.CellEdgeField[ta.wpfloat],
    mass_fl_e: fa.EdgeKField[ta.wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[ta.wpfloat],
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
    n_lev: int32,
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
        mass_fl_e=mass_fl_e,
        z_theta_v_fl_e=z_theta_v_fl_e,
    )

    z_w_expl = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    z_q = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))

    z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q = _stencils_43_44_45(
        z_w_expl=z_w_expl,
        w_nnow=current_w,
        ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
        z_th_ddz_exner_c=z_th_ddz_exner_c,
        z_contr_w_fl_l=z_contr_w_fl_l,
        rho_ic=rho_ic,
        w_concorr_c=contravariant_correction_at_cells_on_half_levels,
        vwind_expl_wgt=vwind_expl_wgt,
        z_beta=z_beta,
        exner_nnow=current_exner,
        rho_nnow=current_rho,
        theta_v_nnow=current_theta_v,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_alpha=z_alpha,
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        z_q=z_q,
        rd=rd,
        cvd=cvd,
        dtime=dtime,
        cpd=cpd,
        nlev=n_lev,
    )

    (next_w, z_contr_w_fl_l) = (
        concat_where(
            dims.KDim == 0,
            (
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            ),
            (next_w, z_contr_w_fl_l),
        )
        if (not l_vert_nested)
        else (next_w, z_contr_w_fl_l)
    )

    (z_rho_expl, z_exner_expl) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=z_contr_w_fl_l,
        exner_pr=exner_pr,
        z_beta=z_beta,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
    )

    z_rho_expl, z_exner_expl = (
        _add_analysis_increments_from_data_assimilation(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        if is_iau_active
        else (z_rho_expl, z_exner_expl)
    )

    z_q, next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_forward_sweep_2(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            z_q=z_q,
            w=next_w,
            dtime=dtime,
            cpd=cpd,
        ),
        (z_q, next_w),
    )

    next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(z_q=z_q, w=next_w),
        next_w,
    )

    return (
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        next_w,
        z_rho_expl,
        z_exner_expl,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_predictor_step_after_solving_w(
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
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
    z_rho_expl: fa.CellKField[ta.wpfloat],
    z_exner_expl: fa.CellKField[ta.wpfloat],
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
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=next_w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
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
        z_beta,
        z_alpha,
        next_w,
        z_rho_expl,
        z_exner_expl,
        next_rho,
        next_exner,
        next_theta_v,
        dwdz_at_cells_on_model_levels,
        exner_dynamical_increment,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_corrector_step_before_solving_w(
    z_contr_w_fl_l: fa.CellKField[ta.wpfloat],
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    geofac_div: fa.CellEdgeField[ta.wpfloat],
    mass_fl_e: fa.EdgeKField[ta.wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[ta.wpfloat],
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
    itime_scheme: int32,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
    cpd: ta.wpfloat,
    is_iau_active: bool,
    n_lev: int32,
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
        mass_fl_e=mass_fl_e,
        z_theta_v_fl_e=z_theta_v_fl_e,
    )

    z_w_expl = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    z_q = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))

    (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q) = (
        _stencils_42_44_45(
            z_w_expl=z_w_expl,
            w_nnow=current_w,
            ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
            ddt_w_adv_ntl2=corrector_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=contravariant_correction_at_cells_on_half_levels,
            vwind_expl_wgt=vwind_expl_wgt,
            z_beta=z_beta,
            exner_nnow=current_exner,
            rho_nnow=current_rho,
            theta_v_nnow=current_theta_v,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            z_q=z_q,
            rd=rd,
            cvd=cvd,
            dtime=dtime,
            cpd=cpd,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            nlev=n_lev,
        )
        if itime_scheme == 4
        else _stencils_43_44_45(
            z_w_expl=z_w_expl,
            w_nnow=current_w,
            ddt_w_adv_ntl1=predictor_vertical_wind_advective_tendency,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=contravariant_correction_at_cells_on_half_levels,
            vwind_expl_wgt=vwind_expl_wgt,
            z_beta=z_beta,
            exner_nnow=current_exner,
            rho_nnow=current_rho,
            theta_v_nnow=current_theta_v,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            z_q=z_q,
            rd=rd,
            cvd=cvd,
            dtime=dtime,
            cpd=cpd,
            nlev=n_lev,
        )
    )

    (next_w, z_contr_w_fl_l) = (
        concat_where(
            dims.KDim == 0,
            (
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            ),
            (next_w, z_contr_w_fl_l),
        )
        if (not l_vert_nested)
        else (next_w, z_contr_w_fl_l)
    )

    (z_rho_expl, z_exner_expl) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=z_contr_w_fl_l,
        exner_pr=exner_pr,
        z_beta=z_beta,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
    )

    z_rho_expl, z_exner_expl = (
        _add_analysis_increments_from_data_assimilation(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        if is_iau_active
        else (z_rho_expl, z_exner_expl)
    )

    z_q, next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_forward_sweep_2(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            z_q=z_q,
            w=next_w,
            dtime=dtime,
            cpd=cpd,
        ),
        (z_q, next_w),
    )

    next_w = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(
            z_q=z_q,
            w=next_w,
        ),
        next_w,
    )

    return (
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        next_w,
        z_rho_expl,
        z_exner_expl,
    )


@gtx.field_operator
def _vertically_implicit_solver_at_corrector_step_after_solving_w(
    z_contr_w_fl_l: fa.CellKField[ta.wpfloat],
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    mass_flx_ic: fa.CellKField[ta.wpfloat],
    vol_flx_ic: fa.CellKField[ta.wpfloat],
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
    z_rho_expl: fa.CellKField[ta.wpfloat],
    z_exner_expl: fa.CellKField[ta.wpfloat],
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
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=next_w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=current_rho,
            theta_v_now=current_theta_v,
            exner_now=current_exner,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        ),
        (next_rho, next_exner, next_theta_v),
    )

    mass_flx_ic, vol_flx_ic = (
        (
            broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
        )
        if (lprep_adv & at_first_substep)
        else (mass_flx_ic, vol_flx_ic)
    )

    mass_flx_ic, vol_flx_ic = concat_where(
        1 <= dims.KDim,
        _update_mass_volume_flux(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=next_w,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
        ),
        (mass_flx_ic, vol_flx_ic),
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
        mass_flx_ic,
        vol_flx_ic,
        exner_dynamical_increment,
    )


@gtx.program
def vertically_implicit_solver_at_predictor_step(
    z_contr_w_fl_l: fa.CellKField[ta.wpfloat],
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    z_rho_expl: fa.CellKField[ta.wpfloat],
    z_exner_expl: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    exner_dynamical_increment: fa.CellKField[ta.vpfloat],
    geofac_div: fa.CellEdgeField[ta.wpfloat],
    mass_fl_e: fa.EdgeKField[ta.wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[ta.wpfloat],
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
    n_lev: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    start_cell_nudging: int32,
    end_cell_local: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_surface_boundary_condtion_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=(
            z_alpha,
            next_w,
            z_contr_w_fl_l,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )

    _vertically_implicit_solver_at_predictor_step_before_solving_w(
        z_contr_w_fl_l=z_contr_w_fl_l,
        z_beta=z_beta,
        z_alpha=z_alpha,
        next_w=next_w,
        geofac_div=geofac_div,
        mass_fl_e=mass_fl_e,
        z_theta_v_fl_e=z_theta_v_fl_e,
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
        n_lev=n_lev,
        out=(
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            next_w,
            z_rho_expl,
            z_exner_expl,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _vertically_implicit_solver_at_predictor_step_after_solving_w(
        z_beta=z_beta,
        z_alpha=z_alpha,
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
        z_rho_expl=z_rho_expl,
        z_exner_expl=z_exner_expl,
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
            z_beta,
            z_alpha,
            next_w,
            z_rho_expl,
            z_exner_expl,
            next_rho,
            next_exner,
            next_theta_v,
            dwdz_at_cells_on_model_levels,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )


@gtx.program
def vertically_implicit_solver_at_corrector_step(
    z_contr_w_fl_l: fa.CellKField[ta.wpfloat],
    z_beta: fa.CellKField[ta.vpfloat],
    z_alpha: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    z_rho_expl: fa.CellKField[ta.wpfloat],
    z_exner_expl: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    mass_flx_ic: fa.CellKField[ta.wpfloat],
    vol_flx_ic: fa.CellKField[ta.wpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    geofac_div: fa.CellEdgeField[ta.wpfloat],
    mass_fl_e: fa.EdgeKField[ta.wpfloat],
    z_theta_v_fl_e: fa.EdgeKField[ta.wpfloat],
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
    itime_scheme: int32,
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
    n_lev: int32,
    jk_start: int32,
    kstart_moist: int32,
    start_cell_nudging: int32,
    end_cell_local: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_surface_boundary_condtion_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=(
            z_alpha,
            next_w,
            z_contr_w_fl_l,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
    _vertically_implicit_solver_at_corrector_step_before_solving_w(
        z_contr_w_fl_l=z_contr_w_fl_l,
        z_beta=z_beta,
        z_alpha=z_alpha,
        next_w=next_w,
        geofac_div=geofac_div,
        mass_fl_e=mass_fl_e,
        z_theta_v_fl_e=z_theta_v_fl_e,
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
        itime_scheme=itime_scheme,
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=dtime,
        rd=rd,
        cvd=cvd,
        cpd=cpd,
        is_iau_active=is_iau_active,
        n_lev=n_lev,
        l_vert_nested=l_vert_nested,
        out=(
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            next_w,
            z_rho_expl,
            z_exner_expl,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _vertically_implicit_solver_at_corrector_step_after_solving_w(
        z_contr_w_fl_l=z_contr_w_fl_l,
        z_beta=z_beta,
        z_alpha=z_alpha,
        next_w=next_w,
        next_rho=next_rho,
        next_exner=next_exner,
        next_theta_v=next_theta_v,
        mass_flx_ic=mass_flx_ic,
        vol_flx_ic=vol_flx_ic,
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
        z_rho_expl=z_rho_expl,
        z_exner_expl=z_exner_expl,
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
            mass_flx_ic,
            vol_flx_ic,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (start_cell_nudging, end_cell_local),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
