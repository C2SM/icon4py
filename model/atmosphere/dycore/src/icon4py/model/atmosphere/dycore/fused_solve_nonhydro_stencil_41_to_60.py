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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.model.atmosphere.dycore.add_analysis_increments_from_data_assimilation import (
    _add_analysis_increments_from_data_assimilation,
)
from icon4py.model.atmosphere.dycore.apply_rayleigh_damping_mechanism import (
    _apply_rayleigh_damping_mechanism,
    _set_w_level_1,
)
from icon4py.model.atmosphere.dycore.compute_divergence_of_fluxes_of_rho_and_theta import (
    _compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_dwdz_for_divergence_damping import (
    _compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.compute_explicit_part_for_rho_and_exner import (
    _compute_explicit_part_for_rho_and_exner,
)
from icon4py.model.atmosphere.dycore.compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    _compute_explicit_vertical_wind_from_advection_and_vertical_wind_density,
)
from icon4py.model.atmosphere.dycore.compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density,
)
from icon4py.model.atmosphere.dycore.compute_results_for_thermodynamic_variables import (
    _compute_results_for_thermodynamic_variables,
)
from icon4py.model.atmosphere.dycore.compute_solver_coefficients_matrix import (
    _compute_solver_coefficients_matrix,
)
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import _copy_cell_kdim_field_to_vp
from icon4py.model.atmosphere.dycore.set_lower_boundary_condition_for_w_and_contravariant_correction import (
    _set_lower_boundary_condition_for_w_and_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_index_to_zero_vp import (
    _set_two_cell_kdim_fields_index_to_zero_vp,
)
from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_to_zero_wp import (
    _set_two_cell_kdim_fields_to_zero_wp,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_back_substitution import (
    _solve_tridiagonal_matrix_for_w_back_substitution_scan,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_forward_sweep import (
    _solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _set_zero_c_k
from icon4py.model.atmosphere.dycore.update_mass_flux import _update_mass_flux
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_solve_nonhydro_stencil_41_to_60_predictor(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_flxdiv_mass: Field[[CellDim, KDim], vpfloat],
    z_flxdiv_theta: Field[[CellDim, KDim], vpfloat],
    z_w_expl: Field[[CellDim, KDim], wpfloat],
    w_nnow: Field[[CellDim, KDim], wpfloat],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], vpfloat],
    z_th_ddz_exner_c: Field[[CellDim, KDim], vpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    z_q: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    ddt_exner_phy: Field[[CellDim, KDim], vpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    z_raylfac: Field[[KDim], wpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    rho: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    z_dwdz_dd: Field[[CellDim, KDim], vpfloat],
    exner_dyn_incr: Field[[CellDim, KDim], vpfloat],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    cvd_o_rd: wpfloat,
    iau_wgt_dyn: wpfloat,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    cpd: wpfloat,
    rayleigh_klemp: int32,
    idiv_method: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    lhdiff_rcf: bool,
    divdamp_type: int32,
    idyn_timestep: int32,
    index_of_damping_layer: int32,
    n_lev: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
):
    vert_idx_1d = vert_idx
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    z_flxdiv_mass, z_flxdiv_theta = (
        where(
            horizontal_lower_0 <= horz_idx < horizontal_upper_0,
            _compute_divergence_of_fluxes_of_rho_and_theta(
                geofac_div=geofac_div,
                mass_fl_e=mass_fl_e,
                z_theta_v_fl_e=z_theta_v_fl_e,
            ),
            (z_flxdiv_mass, z_flxdiv_theta),
        )
        if idiv_method == 1
        else (z_flxdiv_mass, z_flxdiv_theta)
    )

    (z_w_expl, z_contr_w_fl_l) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx >= int32(1)),
        _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
            w_nnow,
            ddt_w_adv_ntl1,
            z_th_ddz_exner_c,
            rho_ic,
            w_concorr_c,
            vwind_expl_wgt,
            dtime,
            cpd,
        ),
        (z_w_expl, z_contr_w_fl_l),
    )
    (z_beta, z_alpha) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _compute_solver_coefficients_matrix(
            exner_nnow,
            rho_nnow,
            theta_v_nnow,
            inv_ddqz_z_full,
            vwind_impl_wgt,
            theta_v_ic,
            rho_ic,
            dtime,
            rd,
            cvd,
        ),
        (z_beta, z_alpha),
    )

    z_alpha, z_q = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _set_two_cell_kdim_fields_index_to_zero_vp(z_alpha, z_q, vert_idx_1d, n_lev, int32(0)),
        (z_alpha, z_q),
    )

    # z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < (n_lev + int32(1))),
    #     _stencils_43_44_45_45b(
    #         z_w_expl=z_w_expl,
    #         w_nnow=w_nnow,
    #         ddt_w_adv_ntl1=ddt_w_adv_ntl1,
    #         z_th_ddz_exner_c=z_th_ddz_exner_c,
    #         z_contr_w_fl_l=z_contr_w_fl_l,
    #         rho_ic=rho_ic,
    #         w_concorr_c=w_concorr_c,
    #         vwind_expl_wgt=vwind_expl_wgt,
    #         z_beta=z_beta,
    #         exner_nnow=exner_nnow,
    #         rho_nnow=rho_nnow,
    #         theta_v_nnow=theta_v_nnow,
    #         inv_ddqz_z_full=inv_ddqz_z_full,
    #         z_alpha=z_alpha,
    #         vwind_impl_wgt=vwind_impl_wgt,
    #         theta_v_ic=theta_v_ic,
    #         z_q=z_q,
    #         k_field=vert_idx_1d,
    #         rd=rd,
    #         cvd=cvd,
    #         dtime=dtime,
    #         cpd=cpd,
    #         nlev=n_lev,
    #     ),
    #     (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
    # )

    w, z_contr_w_fl_l = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx < int32(1)),
            _set_two_cell_kdim_fields_to_zero_wp(),
            (w, z_contr_w_fl_l),
        )
        if not (l_open_ubc & l_vert_nested)
        else (w, z_contr_w_fl_l)
    )

    (w, z_contr_w_fl_l) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx == n_lev),
        _set_lower_boundary_condition_for_w_and_contravariant_correction(w_concorr_c),
        (w, z_contr_w_fl_l),
    )
    # 48 and 49 are identical except for bounds
    (z_rho_expl, z_exner_expl) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _compute_explicit_part_for_rho_and_exner(
            rho_nnow,
            inv_ddqz_z_full,
            z_flxdiv_mass,
            z_contr_w_fl_l,
            exner_pr,
            z_beta,
            z_flxdiv_theta,
            theta_v_ic,
            ddt_exner_phy,
            dtime,
        ),
        (z_rho_expl, z_exner_expl),
    )
    # w, z_contr_w_fl_l, z_rho_expl, z_exner_expl = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper)
    #     & (n_lev <= vert_idx < (n_lev + int32(1))),
    #     _stencils_47_48_49(
    #         w_nnew=w,
    #         z_contr_w_fl_l=z_contr_w_fl_l,
    #         w_concorr_c=w_concorr_c,
    #         z_rho_expl=z_rho_expl,
    #         z_exner_expl=z_exner_expl,
    #         rho_nnow=rho_nnow,
    #         inv_ddqz_z_full=inv_ddqz_z_full,
    #         z_flxdiv_mass=z_flxdiv_mass,
    #         exner_pr=exner_pr,
    #         z_beta=z_beta,
    #         z_flxdiv_theta=z_flxdiv_theta,
    #         theta_v_ic=theta_v_ic,
    #         ddt_exner_phy=ddt_exner_phy,
    #         k_field=vert_idx_1d,
    #         dtime=dtime,
    #         nlev=n_lev,
    #     ),
    #     (w, z_contr_w_fl_l, z_rho_expl, z_exner_expl),
    # )

    z_rho_expl, z_exner_expl = (
        where(
            horizontal_lower_0 <= horz_idx < horizontal_upper_0,
            _add_analysis_increments_from_data_assimilation(
                z_rho_expl=z_rho_expl,
                z_exner_expl=z_exner_expl,
                rho_incr=rho_incr,
                exner_incr=exner_incr,
                iau_wgt_dyn=iau_wgt_dyn,
            ),
            (z_rho_expl, z_exner_expl),
        )
        if is_iau_active
        else (z_rho_expl, z_exner_expl)
    )

    z_q, w = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (int32(1) <= vert_idx),
        _solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            z_q=z_q,
            w=w,
            dtime=dtime,
            cpd=cpd,
        ),
        (z_q, w),
    )

    w = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (int32(1) <= vert_idx),
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(
            z_q=z_q,
            w=w,
        ),
        w,
    )
    w_1 = _set_w_level_1(w, vert_idx_1d)

    w = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0)
            & (int32(1) <= vert_idx < (index_of_damping_layer + int32(1))),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=w,
            ),
            w,
        )
        if rayleigh_type == rayleigh_klemp
        else w
    )

    rho, exner, theta_v = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (jk_start <= vert_idx),
        _compute_results_for_thermodynamic_variables(
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_nnow,
            theta_v_now=theta_v_nnow,
            exner_now=exner_nnow,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        ),
        (rho, exner, theta_v),
    )

    # compute dw/dz for divergence damping term
    z_dwdz_dd = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (kstart_dd3d <= vert_idx),
            _compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=inv_ddqz_z_full,
                w=w,
                w_concorr_c=w_concorr_c,
            ),
            z_dwdz_dd,
        )
        if lhdiff_rcf & (divdamp_type >= 3)
        else z_dwdz_dd
    )

    exner_dyn_incr = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (kstart_moist <= vert_idx),
            _copy_cell_kdim_field_to_vp(field=exner_nnow),
            exner_dyn_incr,
        )
        if idyn_timestep == 1
        else exner_dyn_incr
    )

    return (
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
    )


@field_operator
def _fused_solve_nonhydro_stencil_41_to_60_corrector(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_flxdiv_mass: Field[[CellDim, KDim], vpfloat],
    z_flxdiv_theta: Field[[CellDim, KDim], vpfloat],
    z_w_expl: Field[[CellDim, KDim], wpfloat],
    w_nnow: Field[[CellDim, KDim], wpfloat],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], vpfloat],
    z_th_ddz_exner_c: Field[[CellDim, KDim], vpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    z_q: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    ddt_exner_phy: Field[[CellDim, KDim], vpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    z_raylfac: Field[[KDim], wpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    rho: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    mass_flx_ic: Field[[CellDim, KDim], wpfloat],
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    itime_scheme: int32,
    lprep_adv: bool,
    lclean_mflx: bool,
    r_nsubsteps: wpfloat,
    cvd_o_rd: wpfloat,
    iau_wgt_dyn: wpfloat,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    cpd: wpfloat,
    rayleigh_klemp: int32,
    idiv_method: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    index_of_damping_layer: int32,
    n_lev: int32,
    jk_start: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
):
    vert_idx_1d = vert_idx
    vert_idx = broadcast(vert_idx, (CellDim, KDim))

    # verified for e-9
    z_flxdiv_mass, z_flxdiv_theta = (
        where(
            horizontal_lower_0 <= horz_idx < horizontal_upper_0,
            _compute_divergence_of_fluxes_of_rho_and_theta(
                geofac_div=geofac_div,
                mass_fl_e=mass_fl_e,
                z_theta_v_fl_e=z_theta_v_fl_e,
            ),
            (z_flxdiv_mass, z_flxdiv_theta),
        )
        if idiv_method == 1
        else (z_flxdiv_mass, z_flxdiv_theta)
    )

    (z_w_expl, z_contr_w_fl_l) = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx >= int32(1)),
            _compute_explicit_vertical_wind_from_advection_and_vertical_wind_density(
                w_nnow,
                ddt_w_adv_ntl1,
                ddt_w_adv_ntl2,
                z_th_ddz_exner_c,
                rho_ic,
                w_concorr_c,
                vwind_expl_wgt,
                dtime,
                wgt_nnow_vel,
                wgt_nnew_vel,
                cpd,
            ),
            (z_w_expl, z_contr_w_fl_l),
        )
        if itime_scheme == 4
        else where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx >= int32(1)),
            _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
                w_nnow,
                ddt_w_adv_ntl1,
                z_th_ddz_exner_c,
                rho_ic,
                w_concorr_c,
                vwind_expl_wgt,
                dtime,
                cpd,
            ),
            (z_w_expl, z_contr_w_fl_l),
        )
    )

    (z_beta, z_alpha) = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
            _compute_solver_coefficients_matrix(
                exner_nnow,
                rho_nnow,
                theta_v_nnow,
                inv_ddqz_z_full,
                vwind_impl_wgt,
                theta_v_ic,
                rho_ic,
                dtime,
                rd,
                cvd,
            ),
            (z_beta, z_alpha),
        )
        if itime_scheme == 4
        else (z_beta, z_alpha)
    )

    z_alpha, z_q = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
            _set_two_cell_kdim_fields_index_to_zero_vp(z_alpha, z_q, vert_idx_1d, n_lev, int32(0)),
            (z_alpha, z_q),
        )
        if itime_scheme == 4
        else (z_alpha, z_q)
    )

    # (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q) = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < n_lev + 1),
    #     _stencils_42_44_45_45b(
    #         z_w_expl=z_w_expl,
    #         w_nnow=w_nnow,
    #         ddt_w_adv_ntl1=ddt_w_adv_ntl1,
    #         ddt_w_adv_ntl2=ddt_w_adv_ntl2,
    #         z_th_ddz_exner_c=z_th_ddz_exner_c,
    #         z_contr_w_fl_l=z_contr_w_fl_l,
    #         rho_ic=rho_ic,
    #         w_concorr_c=w_concorr_c,
    #         vwind_expl_wgt=vwind_expl_wgt,
    #         z_beta=z_beta,
    #         exner_nnow=exner_nnow,
    #         rho_nnow=rho_nnow,
    #         theta_v_nnow=theta_v_nnow,
    #         inv_ddqz_z_full=inv_ddqz_z_full,
    #         z_alpha=z_alpha,
    #         vwind_impl_wgt=vwind_impl_wgt,
    #         theta_v_ic=theta_v_ic,
    #         z_q=z_q,
    #         k_field=vert_idx_1d,
    #         rd=rd,
    #         cvd=cvd,
    #         dtime=dtime,
    #         cpd=cpd,
    #         wgt_nnow_vel=wgt_nnow_vel,
    #         wgt_nnew_vel=wgt_nnew_vel,
    #         nlev=n_lev,
    #     ),
    #     (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
    # )
    # (z_w_expl, z_contr_w_fl_l) = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx >= int32(1)),
    #     _mo_solve_nonhydro_stencil_43(
    #         w_nnow,
    #         ddt_w_adv_ntl1,
    #         z_th_ddz_exner_c,
    #         rho_ic,
    #         w_concorr_c,
    #         vwind_expl_wgt,
    #         dtime,
    #         cpd,
    #     ),
    #     (z_w_expl, z_contr_w_fl_l),
    # )
    # (z_beta, z_alpha) = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper),
    #     _mo_solve_nonhydro_stencil_44(
    #         exner_nnow,
    #         rho_nnow,
    #         theta_v_nnow,
    #         inv_ddqz_z_full,
    #         vwind_impl_wgt,
    #         theta_v_ic,
    #         rho_ic,
    #         dtime,
    #         rd,
    #         cvd,
    #     ),
    #     (z_beta, z_alpha),
    # )
    # z_alpha = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx == n_lev),
    #     _mo_solve_nonhydro_stencil_45(),
    #     z_alpha,
    # )
    # z_q = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx == int32(0)),
    #     _mo_solve_nonhydro_stencil_45_b(),
    #     z_q,
    # )

    # (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q) = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < n_lev + 1),
    #     _stencils_43_44_45_45b(
    #         z_w_expl=z_w_expl,
    #         w_nnow=w_nnow,
    #         ddt_w_adv_ntl1=ddt_w_adv_ntl1,
    #         z_th_ddz_exner_c=z_th_ddz_exner_c,
    #         z_contr_w_fl_l=z_contr_w_fl_l,
    #         rho_ic=rho_ic,
    #         w_concorr_c=w_concorr_c,
    #         vwind_expl_wgt=vwind_expl_wgt,
    #         z_beta=z_beta,
    #         exner_nnow=exner_nnow,
    #         rho_nnow=rho_nnow,
    #         theta_v_nnow=theta_v_nnow,
    #         inv_ddqz_z_full=inv_ddqz_z_full,
    #         z_alpha=z_alpha,
    #         vwind_impl_wgt=vwind_impl_wgt,
    #         theta_v_ic=theta_v_ic,
    #         z_q=z_q,
    #         k_field=vert_idx_1d,
    #         rd=rd,
    #         cvd=cvd,
    #         dtime=dtime,
    #         cpd=cpd,
    #         nlev=n_lev,
    #     ),
    #     (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
    # )

    w, z_contr_w_fl_l = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx < int32(1)),
            _set_two_cell_kdim_fields_index_to_zero_vp(
                w, z_contr_w_fl_l, vert_idx_1d, int32(0), int32(0)
            ),
            (w, z_contr_w_fl_l),
        )
        if not (l_open_ubc & l_vert_nested)
        else (w, z_contr_w_fl_l)
    )

    (w, z_contr_w_fl_l) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx == n_lev),
        _set_lower_boundary_condition_for_w_and_contravariant_correction(w_concorr_c),
        (w, z_contr_w_fl_l),
    )
    # 48 and 49 are identical except for bounds
    (z_rho_expl, z_exner_expl) = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _compute_explicit_part_for_rho_and_exner(
            rho_nnow,
            inv_ddqz_z_full,
            z_flxdiv_mass,
            z_contr_w_fl_l,
            exner_pr,
            z_beta,
            z_flxdiv_theta,
            theta_v_ic,
            ddt_exner_phy,
            dtime,
        ),
        (z_rho_expl, z_exner_expl),
    )
    # w, z_contr_w_fl_l, z_rho_expl, z_exner_expl = where(
    #     (horizontal_lower <= horz_idx < horizontal_upper)
    #     & (n_lev <= vert_idx < (n_lev + int32(1))),
    #     _stencils_47_48_49(
    #         w_nnew=w,
    #         z_contr_w_fl_l=z_contr_w_fl_l,
    #         w_concorr_c=w_concorr_c,
    #         z_rho_expl=z_rho_expl,
    #         z_exner_expl=z_exner_expl,
    #         rho_nnow=rho_nnow,
    #         inv_ddqz_z_full=inv_ddqz_z_full,
    #         z_flxdiv_mass=z_flxdiv_mass,
    #         exner_pr=exner_pr,
    #         z_beta=z_beta,
    #         z_flxdiv_theta=z_flxdiv_theta,
    #         theta_v_ic=theta_v_ic,
    #         ddt_exner_phy=ddt_exner_phy,
    #         k_field=vert_idx_1d,
    #         dtime=dtime,
    #         nlev=n_lev,
    #     ),
    #     (w, z_contr_w_fl_l, z_rho_expl, z_exner_expl),
    # )

    z_rho_expl, z_exner_expl = (
        where(
            horizontal_lower_0 <= horz_idx < horizontal_upper_0,
            _add_analysis_increments_from_data_assimilation(
                z_rho_expl=z_rho_expl,
                z_exner_expl=z_exner_expl,
                rho_incr=rho_incr,
                exner_incr=exner_incr,
                iau_wgt_dyn=iau_wgt_dyn,
            ),
            (z_rho_expl, z_exner_expl),
        )
        if is_iau_active
        else (z_rho_expl, z_exner_expl)
    )

    z_q, w = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (int32(1) <= vert_idx),
        _solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            z_q=z_q,
            w=w,
            dtime=dtime,
            cpd=cpd,
        ),
        (z_q, w),
    )

    w = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (int32(1) <= vert_idx),
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(
            z_q=z_q,
            w=w,
        ),
        w,
    )
    w_1 = _set_w_level_1(w, vert_idx_1d)

    w = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0)
            & (int32(1) <= vert_idx < (index_of_damping_layer + int32(1))),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=w,
            ),
            w,
        )
        if rayleigh_type == rayleigh_klemp
        else w
    )

    rho, exner, theta_v = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (jk_start <= vert_idx),
        _compute_results_for_thermodynamic_variables(
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_nnow,
            theta_v_now=theta_v_nnow,
            exner_now=exner_nnow,
            dtime=dtime,
            cvd_o_rd=cvd_o_rd,
        ),
        (rho, exner, theta_v),
    )

    mass_flx_ic = (
        where(
            (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
            _set_zero_c_k(),
            mass_flx_ic,
        )
        if (lprep_adv & lclean_mflx)
        else mass_flx_ic
    )

    mass_flx_ic = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _update_mass_flux(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
        ),
        mass_flx_ic,
    )
    return (
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        mass_flx_ic,
    )


@field_operator
def _fused_solve_nonhydro_stencil_41_to_60(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    w_nnow: Field[[CellDim, KDim], wpfloat],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], vpfloat],
    z_th_ddz_exner_c: Field[[CellDim, KDim], vpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    ddt_exner_phy: Field[[CellDim, KDim], vpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    z_raylfac: Field[[KDim], wpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    z_flxdiv_mass: Field[[CellDim, KDim], vpfloat],
    z_flxdiv_theta: Field[[CellDim, KDim], vpfloat],
    z_w_expl: Field[[CellDim, KDim], wpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    z_q: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    rho: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    z_dwdz_dd: Field[[CellDim, KDim], vpfloat],
    exner_dyn_incr: Field[[CellDim, KDim], vpfloat],
    mass_flx_ic: Field[[CellDim, KDim], wpfloat],
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    itime_scheme: int32,
    lprep_adv: bool,
    lclean_mflx: bool,
    r_nsubsteps: wpfloat,
    cvd_o_rd: wpfloat,
    iau_wgt_dyn: wpfloat,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    cpd: wpfloat,
    rayleigh_klemp: int32,
    idiv_method: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    lhdiff_rcf: bool,
    divdamp_type: int32,
    idyn_timestep: int32,
    index_of_damping_layer: int32,
    n_lev: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
    istep: int32,
):
    (
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        w,
        z_rho_expl,
        z_exner_expl,
        z_q,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
    ) = (
        _fused_solve_nonhydro_stencil_41_to_60_predictor(
            geofac_div=geofac_div,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_flxdiv_mass=z_flxdiv_mass,
            z_flxdiv_theta=z_flxdiv_theta,
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            z_beta=z_beta,
            exner_nnow=exner_nnow,
            rho_nnow=rho_nnow,
            theta_v_nnow=theta_v_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            z_q=z_q,
            w=w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            exner_pr=exner_pr,
            ddt_exner_phy=ddt_exner_phy,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            ddqz_z_half=ddqz_z_half,
            z_raylfac=z_raylfac,
            exner_ref_mc=exner_ref_mc,
            rho=rho,
            exner=exner,
            theta_v=theta_v,
            z_dwdz_dd=z_dwdz_dd,
            exner_dyn_incr=exner_dyn_incr,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            cvd_o_rd=cvd_o_rd,
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
            cpd=cpd,
            rayleigh_klemp=rayleigh_klemp,
            idiv_method=idiv_method,
            l_open_ubc=l_open_ubc,
            l_vert_nested=l_vert_nested,
            is_iau_active=is_iau_active,
            rayleigh_type=rayleigh_type,
            lhdiff_rcf=lhdiff_rcf,
            divdamp_type=divdamp_type,
            idyn_timestep=idyn_timestep,
            index_of_damping_layer=index_of_damping_layer,
            n_lev=n_lev,
            jk_start=jk_start,
            kstart_dd3d=kstart_dd3d,
            kstart_moist=kstart_moist,
            horizontal_lower_0=horizontal_lower_0,
            horizontal_upper_0=horizontal_upper_0,
        )
        if istep == 1
        else (
            z_flxdiv_mass,
            z_flxdiv_theta,
            z_w_expl,
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            w,
            z_rho_expl,
            z_exner_expl,
            z_q,
            rho,
            exner,
            theta_v,
            z_dwdz_dd,
            exner_dyn_incr,
        )
    )

    (
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        mass_flx_ic,
    ) = (
        _fused_solve_nonhydro_stencil_41_to_60_corrector(
            geofac_div=geofac_div,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_flxdiv_mass=z_flxdiv_mass,
            z_flxdiv_theta=z_flxdiv_theta,
            z_w_expl=z_w_expl,
            w_nnow=w_nnow,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            w_concorr_c=w_concorr_c,
            vwind_expl_wgt=vwind_expl_wgt,
            z_beta=z_beta,
            exner_nnow=exner_nnow,
            rho_nnow=rho_nnow,
            theta_v_nnow=theta_v_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            z_q=z_q,
            w=w,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            exner_pr=exner_pr,
            ddt_exner_phy=ddt_exner_phy,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            ddqz_z_half=ddqz_z_half,
            z_raylfac=z_raylfac,
            exner_ref_mc=exner_ref_mc,
            rho=rho,
            exner=exner,
            theta_v=theta_v,
            mass_flx_ic=mass_flx_ic,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            cvd_o_rd=cvd_o_rd,
            iau_wgt_dyn=iau_wgt_dyn,
            rd=rd,
            cvd=cvd,
            dtime=dtime,
            cpd=cpd,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            n_lev=n_lev,
            idiv_method=idiv_method,
            itime_scheme=itime_scheme,
            l_open_ubc=l_open_ubc,
            l_vert_nested=l_vert_nested,
            is_iau_active=is_iau_active,
            index_of_damping_layer=index_of_damping_layer,
            rayleigh_klemp=rayleigh_klemp,
            rayleigh_type=rayleigh_type,
            jk_start=jk_start,
            lprep_adv=lprep_adv,
            lclean_mflx=lclean_mflx,
            r_nsubsteps=r_nsubsteps,
            horizontal_lower_0=horizontal_lower_0,
            horizontal_upper_0=horizontal_upper_0,
        )
        if istep > 1
        else (
            z_flxdiv_mass,
            z_flxdiv_theta,
            z_w_expl,
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            z_q,
            w,
            z_rho_expl,
            z_exner_expl,
            rho,
            exner,
            theta_v,
            mass_flx_ic,
        )
    )

    return (
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
        mass_flx_ic,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_41_to_60(
    geofac_div: Field[[CEDim], wpfloat],
    mass_fl_e: Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], wpfloat],
    w_nnow: Field[[CellDim, KDim], wpfloat],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], vpfloat],
    z_th_ddz_exner_c: Field[[CellDim, KDim], vpfloat],
    rho_ic: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    exner_nnow: Field[[CellDim, KDim], wpfloat],
    rho_nnow: Field[[CellDim, KDim], wpfloat],
    theta_v_nnow: Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    ddt_exner_phy: Field[[CellDim, KDim], vpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    z_raylfac: Field[[KDim], wpfloat],
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    z_flxdiv_mass: Field[[CellDim, KDim], vpfloat],
    z_flxdiv_theta: Field[[CellDim, KDim], vpfloat],
    z_w_expl: Field[[CellDim, KDim], wpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    z_q: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    rho: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    z_dwdz_dd: Field[[CellDim, KDim], vpfloat],
    exner_dyn_incr: Field[[CellDim, KDim], vpfloat],
    mass_flx_ic: Field[[CellDim, KDim], wpfloat],
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    itime_scheme: int32,
    lprep_adv: bool,
    lclean_mflx: bool,
    r_nsubsteps: wpfloat,
    cvd_o_rd: wpfloat,
    iau_wgt_dyn: wpfloat,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    cpd: wpfloat,
    rayleigh_klemp: int32,
    idiv_method: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: int32,
    lhdiff_rcf: bool,
    divdamp_type: int32,
    idyn_timestep: int32,
    index_of_damping_layer: int32,
    n_lev: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    istep: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_solve_nonhydro_stencil_41_to_60(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        w_nnow,
        ddt_w_adv_ntl1,
        ddt_w_adv_ntl2,
        z_th_ddz_exner_c,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_ic,
        exner_pr,
        ddt_exner_phy,
        rho_incr,
        exner_incr,
        ddqz_z_half,
        z_raylfac,
        exner_ref_mc,
        horz_idx,
        vert_idx,
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
        mass_flx_ic,
        wgt_nnow_vel,
        wgt_nnew_vel,
        itime_scheme,
        lprep_adv,
        lclean_mflx,
        r_nsubsteps,
        cvd_o_rd,
        iau_wgt_dyn,
        dtime,
        rd,
        cvd,
        cpd,
        rayleigh_klemp,
        idiv_method,
        l_open_ubc,
        l_vert_nested,
        is_iau_active,
        rayleigh_type,
        lhdiff_rcf,
        divdamp_type,
        idyn_timestep,
        index_of_damping_layer,
        n_lev,
        jk_start,
        kstart_dd3d,
        kstart_moist,
        horizontal_lower_0,
        horizontal_upper_0,
        istep,
        out=(
            z_flxdiv_mass,
            z_flxdiv_theta,
            z_w_expl,
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            z_q,
            w,
            z_rho_expl,
            z_exner_expl,
            rho,
            exner,
            theta_v,
            z_dwdz_dd,
            exner_dyn_incr,
            mass_flx_ic,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )

    _fused_solve_nonhydro_stencil_41_to_60(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        w_nnow,
        ddt_w_adv_ntl1,
        ddt_w_adv_ntl2,
        z_th_ddz_exner_c,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_ic,
        exner_pr,
        ddt_exner_phy,
        rho_incr,
        exner_incr,
        ddqz_z_half,
        z_raylfac,
        exner_ref_mc,
        horz_idx,
        vert_idx,
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        z_contr_w_fl_l,
        z_beta,
        z_alpha,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
        mass_flx_ic,
        wgt_nnow_vel,
        wgt_nnew_vel,
        itime_scheme,
        lprep_adv,
        lclean_mflx,
        r_nsubsteps,
        cvd_o_rd,
        iau_wgt_dyn,
        dtime,
        rd,
        cvd,
        cpd,
        rayleigh_klemp,
        idiv_method,
        l_open_ubc,
        l_vert_nested,
         is_iau_active,
        rayleigh_type,
        lhdiff_rcf,
        divdamp_type,
        idyn_timestep,
        index_of_damping_layer,
        n_lev,
        jk_start,
        kstart_dd3d,
        kstart_moist,
        horizontal_lower_0,
        horizontal_upper_0,
        istep,
        out=(
            z_flxdiv_mass,
            z_flxdiv_theta,
            z_w_expl,
            z_contr_w_fl_l,
            z_beta,
            z_alpha,
            z_q,
            w,
            z_rho_expl,
            z_exner_expl,
            rho,
            exner,
            theta_v,
            z_dwdz_dd,
            exner_dyn_incr,
            mass_flx_ic,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
