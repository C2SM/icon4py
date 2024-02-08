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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.fused_solve_nonhydro_stencil_41_to_60 import (
    fused_solve_nonhydro_stencil_41_to_60,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field

from .test_add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation_numpy,
)
from .test_apply_rayleigh_damping_mechanism import apply_rayleigh_damping_mechanism_numpy
from .test_compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta_numpy,
)
from .test_compute_dwdz_for_divergence_damping import compute_dwdz_for_divergence_damping_numpy
from .test_compute_explicit_part_for_rho_and_exner import (
    compute_explicit_part_for_rho_and_exner_numpy,
)
from .test_compute_explicit_vertical_wind_from_advection_and_vertical_wind_density import (
    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy,
)
from .test_compute_explicit_vertical_wind_speed_and_vertical_wind_times_density import (
    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy,
)
from .test_compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables_numpy,
)
from .test_compute_solver_coefficients_matrix import compute_solver_coefficients_matrix_numpy
from .test_copy_cell_kdim_field_to_vp import copy_cell_kdim_field_to_vp_numpy
from .test_set_lower_boundary_condition_for_w_and_contravariant_correction import (
    set_lower_boundary_condition_for_w_and_contravariant_correction_numpy,
)
from .test_set_two_cell_kdim_fields_index_to_zero_vp import (
    set_two_cell_kdim_fields_index_to_zero_vp_numpy,
)
from .test_set_two_cell_kdim_fields_to_zero_wp import set_two_cell_kdim_fields_to_zero_wp_numpy
from .test_solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution_numpy,
)
from .test_solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep_numpy,
)
from .test_update_mass_flux import update_mass_flux_numpy


class TestFusedMoSolveNonHydroStencil41To60(StencilTest):
    PROGRAM = fused_solve_nonhydro_stencil_41_to_60
    OUTPUTS = (
        "z_flxdiv_mass",
        "z_flxdiv_theta",
        "z_w_expl",
        "z_contr_w_fl_l",
        "z_beta",
        "z_alpha",
        "z_q",
        "w",
        "z_rho_expl",
        "z_exner_expl",
        "rho",
        "exner",
        "theta_v",
        "z_dwdz_dd",
        "exner_dyn_incr",
        "mass_flx_ic",
    )

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        grid,
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        z_flxdiv_mass,
        z_flxdiv_theta,
        z_w_expl,
        w_nnow,
        ddt_w_adv_ntl1,
        ddt_w_adv_ntl2,
        z_th_ddz_exner_c,
        z_contr_w_fl_l,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        z_beta,
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        z_alpha,
        vwind_impl_wgt,
        theta_v_ic,
        z_q,
        w,
        z_rho_expl,
        z_exner_expl,
        exner_pr,
        ddt_exner_phy,
        rho_incr,
        exner_incr,
        ddqz_z_half,
        z_raylfac,
        exner_ref_mc,
        rho,
        exner,
        theta_v,
        z_dwdz_dd,
        exner_dyn_incr,
        horz_idx,
        vert_idx,
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
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
    ):
        horz_idx = horz_idx[:, np.newaxis]
        if istep == 1:
            if idiv_method == 1:
                # verified for e-9
                z_flxdiv_mass, z_flxdiv_theta = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                        grid=grid,
                        geofac_div=geofac_div,
                        mass_fl_e=mass_fl_e,
                        z_theta_v_fl_e=z_theta_v_fl_e,
                    ),
                    (z_flxdiv_mass, z_flxdiv_theta),
                )

            (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= int32(1))
                & (vert_idx < n_lev),
                compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
                    grid=grid,
                    w_nnow=w_nnow,
                    ddt_w_adv_ntl1=ddt_w_adv_ntl1,
                    z_th_ddz_exner_c=z_th_ddz_exner_c,
                    rho_ic=rho_ic[:, :n_lev],
                    w_concorr_c=w_concorr_c[:, :n_lev],
                    vwind_expl_wgt=vwind_expl_wgt,
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )
            (z_beta, z_alpha[:, :n_lev]) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= int32(0))
                & (vert_idx < n_lev),
                compute_solver_coefficients_matrix_numpy(
                    grid=grid,
                    exner_nnow=exner_nnow,
                    rho_nnow=rho_nnow,
                    theta_v_nnow=theta_v_nnow,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_ic[:, :n_lev],
                    rho_ic=rho_ic[:, :n_lev],
                    dtime=dtime,
                    rd=rd,
                    cvd=cvd,
                ),
                (z_beta, z_alpha[:, :n_lev]),
            )
            z_alpha[:, :n_lev], z_q = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx == n_lev),
                set_two_cell_kdim_fields_index_to_zero_vp_numpy(
                    grid=grid,
                    field_index_to_zero_1=z_alpha[:, :n_lev],
                    field_index_to_zero_2=z_q,
                    k=vert_idx,
                    k1=n_lev,
                    k2=int32(0),
                ),
                (z_alpha[:, :n_lev], z_q),
            )
            # z_q = np.where(
            #     (horizontal_lower <= horz_idx)
            #     & (horz_idx < horizontal_upper)
            #     & (vert_idx == int32(0)),
            #     0,
            #     z_q,
            # )

            if not (l_open_ubc and l_vert_nested):
                w[:, :n_lev], z_contr_w_fl_l[:, :n_lev] = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx == 0),
                    set_two_cell_kdim_fields_to_zero_wp_numpy(
                        grid=grid,
                        cell_kdim_field_to_zero_wp_1=w[:, :n_lev],
                        cell_kdim_field_to_zero_wp_2=z_contr_w_fl_l[:, :n_lev],
                    ),
                    (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
                )

            (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx == n_lev),
                set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
                    grid=grid,
                    w_concorr_c=w_concorr_c[:, :n_lev],
                    z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                ),
                (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )
            # 48 and 49 are identical except for bounds
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= int32(0))
                & (vert_idx < n_lev),
                compute_explicit_part_for_rho_and_exner_numpy(
                    grid=grid,
                    rho_nnow=rho_nnow,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    z_flxdiv_mass=z_flxdiv_mass,
                    z_contr_w_fl_l=z_contr_w_fl_l,
                    exner_pr=exner_pr,
                    z_beta=z_beta,
                    z_flxdiv_theta=z_flxdiv_theta,
                    theta_v_ic=theta_v_ic,
                    ddt_exner_phy=ddt_exner_phy,
                    dtime=dtime,
                ),
                (z_rho_expl, z_exner_expl),
            )

            if is_iau_active:
                (z_rho_expl, z_exner_expl) = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    add_analysis_increments_from_data_assimilation_numpy(
                        grid=grid,
                        z_rho_expl=z_rho_expl,
                        z_exner_expl=z_exner_expl,
                        rho_incr=rho_incr,
                        exner_incr=exner_incr,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    (z_rho_expl, z_exner_expl),
                )
            z_q, w[:, :n_lev] = np.where(
                (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0) & (vert_idx >= 1),
                solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_ic[:, :n_lev],
                    ddqz_z_half=ddqz_z_half,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    z_w_expl=z_w_expl,
                    z_exner_expl=z_exner_expl,
                    z_q_ref=z_q,
                    w_ref=w[:, :n_lev],
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_q, w[:, :n_lev]),
            )

            w[:, :n_lev] = np.where(
                (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0) & (vert_idx >= 1),
                solve_tridiagonal_matrix_for_w_back_substitution_numpy(
                    grid=grid,
                    z_q=z_q,
                    w=w[:, :n_lev],
                ),
                w[:, :n_lev],
            )

            w_1 = w[:, :1]
            if rayleigh_type == rayleigh_klemp:
                w[:, :n_lev] = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= 1)
                    & (vert_idx < (index_of_damping_layer + 1)),
                    apply_rayleigh_damping_mechanism_numpy(
                        grid=grid,
                        z_raylfac=z_raylfac,
                        w_1=w_1,
                        w=w[:, :n_lev],
                    ),
                    w[:, :n_lev],
                )

            rho, exner, theta_v = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= jk_start),
                compute_results_for_thermodynamic_variables_numpy(
                    grid=grid,
                    z_rho_expl=z_rho_expl,
                    vwind_impl_wgt=vwind_impl_wgt,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    rho_ic=rho_ic,
                    w=w,
                    z_exner_expl=z_exner_expl,
                    exner_ref_mc=exner_ref_mc,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    rho_now=rho,
                    theta_v_now=theta_v,
                    exner_now=exner,
                    dtime=dtime,
                    cvd_o_rd=cvd_o_rd,
                ),
                (rho, exner, theta_v),
            )

            # compute dw/dz for divergence damping term
            if lhdiff_rcf and divdamp_type >= 3:
                z_dwdz_dd = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= kstart_dd3d),
                    compute_dwdz_for_divergence_damping_numpy(
                        grid=grid, inv_ddqz_z_full=inv_ddqz_z_full, w=w, w_concorr_c=w_concorr_c
                    ),
                    z_dwdz_dd,
                )

            if idyn_timestep == 1:
                exner_dyn_incr = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= kstart_moist),
                    copy_cell_kdim_field_to_vp_numpy(field=exner),
                    exner_dyn_incr,
                )

        else:

            if idiv_method == 1:
                # verified for e-9
                z_flxdiv_mass, z_flxdiv_theta = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                        grid=grid,
                        geofac_div=geofac_div,
                        mass_fl_e=mass_fl_e,
                        z_theta_v_fl_e=z_theta_v_fl_e,
                    ),
                    (z_flxdiv_mass, z_flxdiv_theta),
                )

            if itime_scheme == 4:

                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= int32(1))
                    & (vert_idx < n_lev),
                    compute_explicit_vertical_wind_from_advection_and_vertical_wind_density_numpy(
                        grid=grid,
                        w_nnow=w_nnow,
                        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
                        ddt_w_adv_ntl2=ddt_w_adv_ntl2,
                        z_th_ddz_exner_c=z_th_ddz_exner_c,
                        rho_ic=rho_ic[:, :n_lev],
                        w_concorr_c=w_concorr_c[:, :n_lev],
                        vwind_expl_wgt=vwind_expl_wgt,
                        dtime=dtime,
                        wgt_nnow_vel=wgt_nnow_vel,
                        wgt_nnew_vel=wgt_nnew_vel,
                        cpd=cpd,
                    ),
                    (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
                )

                (z_beta, z_alpha[:, :n_lev]) = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= int32(0))
                    & (vert_idx < n_lev),
                    compute_solver_coefficients_matrix_numpy(
                        grid=grid,
                        exner_nnow=exner_nnow,
                        rho_nnow=rho_nnow,
                        theta_v_nnow=theta_v_nnow,
                        inv_ddqz_z_full=inv_ddqz_z_full,
                        vwind_impl_wgt=vwind_impl_wgt,
                        theta_v_ic=theta_v_ic[:, :n_lev],
                        rho_ic=rho_ic[:, :n_lev],
                        dtime=dtime,
                        rd=rd,
                        cvd=cvd,
                    ),
                    (z_beta, z_alpha[:, :n_lev]),
                )
                z_alpha[:, :n_lev], z_q = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    set_two_cell_kdim_fields_index_to_zero_vp_numpy(
                        grid=grid,
                        field_index_to_zero_1=z_alpha[:, :n_lev],
                        field_index_to_zero_2=z_q,
                        k=vert_idx,
                        k1=n_lev,
                        k2=0,
                    ),
                    (z_alpha[:, :n_lev], z_q),
                )

                # z_q = np.where(
                #     (horizontal_lower <= horz_idx)
                #     & (horz_idx < horizontal_upper)
                #     & (vert_idx == 0),
                #     0,
                #     z_q,
                # )

            else:
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                    (vert_idx >= int32(1)) & (vert_idx < n_lev),
                    compute_explicit_vertical_wind_speed_and_vertical_wind_times_density_numpy(
                        grid=grid,
                        w_nnow=w_nnow,
                        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
                        z_th_ddz_exner_c=z_th_ddz_exner_c,
                        rho_ic=rho_ic[:, :n_lev],
                        w_concorr_c=w_concorr_c[:, :n_lev],
                        vwind_expl_wgt=vwind_expl_wgt,
                        dtime=dtime,
                        cpd=cpd,
                    ),
                    (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
                )
                (z_beta, z_alpha) = np.where(
                    (vert_idx >= int32(0)) & (vert_idx < n_lev),
                    compute_solver_coefficients_matrix_numpy(
                        grid=grid,
                        exner_nnow=exner_nnow,
                        rho_nnow=rho_nnow,
                        theta_v_nnow=theta_v_nnow,
                        inv_ddqz_z_full=inv_ddqz_z_full,
                        vwind_impl_wgt=vwind_impl_wgt,
                        theta_v_ic=theta_v_ic,
                        rho_ic=rho_ic,
                        dtime=dtime,
                        rd=rd,
                        cvd=cvd,
                    ),
                    (z_beta, z_alpha),
                )
                z_alpha[:, :n_lev], z_q = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    set_two_cell_kdim_fields_index_to_zero_vp_numpy(
                        grid=grid,
                        field_index_to_zero_1=z_alpha[:, :n_lev],
                        field_index_to_zero_2=z_q,
                        k=vert_idx,
                        k1=n_lev,
                        k2=0,
                    ),
                    (z_alpha[:, :n_lev], z_q),
                )

                # z_alpha = np.where(
                #     (horizontal_lower <= horz_idx)
                #     & (horz_idx < horizontal_upper)
                #     & (vert_idx == n_lev),
                #     set_two_cell_kdim_fields_index_to_zero_vp_numpy(grid=grid, z_alpha=z_alpha),
                #     z_alpha,
                # )
                # z_q = np.where(
                #     (horizontal_lower <= horz_idx)
                #     & (horz_idx < horizontal_upper)
                #     & (vert_idx == int32(0)),
                #     0,
                #     z_q,
                # )

            if not (l_open_ubc and not l_vert_nested):
                w[:, :n_lev], z_contr_w_fl_l[:, :n_lev] = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx == 0),
                    set_two_cell_kdim_fields_to_zero_wp_numpy(
                        grid=grid,
                        w_nnew=w[:, :n_lev],
                        z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                    ),
                    (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
                )

            (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx == n_lev),
                set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
                    grid, w_concorr_c[:, :n_lev], z_contr_w_fl_l[:, :n_lev]
                ),
                (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )
            # 48 and 49 are identical except for bounds
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= int32(0))
                & (vert_idx < n_lev),
                compute_explicit_part_for_rho_and_exner_numpy(
                    grid=grid,
                    rho_nnow=rho_nnow,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    z_flxdiv_mass=z_flxdiv_mass,
                    z_contr_w_fl_l=z_contr_w_fl_l,
                    exner_pr=exner_pr,
                    z_beta=z_beta,
                    z_flxdiv_theta=z_flxdiv_theta,
                    theta_v_ic=theta_v_ic,
                    ddt_exner_phy=ddt_exner_phy,
                    dtime=dtime,
                ),
                (z_rho_expl, z_exner_expl),
            )

            if is_iau_active:
                (z_rho_expl, z_exner_expl) = np.where(
                    (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                    add_analysis_increments_from_data_assimilation_numpy(
                        grid=grid,
                        z_rho_expl=z_rho_expl,
                        z_exner_expl=z_exner_expl,
                        rho_incr=rho_incr,
                        exner_incr=exner_incr,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    (z_rho_expl, z_exner_expl),
                )

            z_q, w[:, :n_lev] = np.where(
                (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0) & (vert_idx >= 1),
                solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_ic[:, :n_lev],
                    ddqz_z_half=ddqz_z_half,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    z_w_expl=z_w_expl,
                    z_exner_expl=z_exner_expl,
                    z_q_ref=z_q,
                    w_ref=w[:, :n_lev],
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_q, w[:, :n_lev]),
            )

            w[:, :n_lev] = np.where(
                (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0) & (vert_idx >= 1),
                solve_tridiagonal_matrix_for_w_back_substitution_numpy(
                    grid=grid,
                    z_q=z_q,
                    w=w[:, :n_lev],
                ),
                w[:, :n_lev],
            )

            w_1 = w[:, :1]
            if rayleigh_type == rayleigh_klemp:
                w[:, :n_lev] = np.where(
                    (horizontal_lower_0 <= horz_idx)
                    & (horz_idx < horizontal_upper_0)
                    & (vert_idx >= 1)
                    & (vert_idx < (index_of_damping_layer + 1)),
                    apply_rayleigh_damping_mechanism_numpy(
                        grid=grid,
                        z_raylfac=z_raylfac,
                        w_1=w_1,
                        w=w[:, :n_lev],
                    ),
                    w[:, :n_lev],
                )

            rho, exner, theta_v = np.where(
                (horizontal_lower_0 <= horz_idx)
                & (horz_idx < horizontal_upper_0)
                & (vert_idx >= jk_start),
                compute_results_for_thermodynamic_variables_numpy(
                    grid=grid,
                    z_rho_expl=z_rho_expl,
                    vwind_impl_wgt=vwind_impl_wgt,
                    inv_ddqz_z_full=inv_ddqz_z_full,
                    rho_ic=rho_ic,
                    w=w,
                    z_exner_expl=z_exner_expl,
                    exner_ref_mc=exner_ref_mc,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    rho_now=rho,
                    theta_v_now=theta_v,
                    exner_now=exner,
                    dtime=dtime,
                    cvd_o_rd=cvd_o_rd,
                ),
                (rho, exner, theta_v),
            )

            if lprep_adv:
                if lclean_mflx:
                    mass_flx_ic = np.zeros_like(exner)

            mass_flx_ic = np.where(
                (horizontal_lower_0 <= horz_idx) & (horz_idx < horizontal_upper_0),
                update_mass_flux_numpy(
                    grid=grid,
                    z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                    rho_ic=rho_ic[:, :n_lev],
                    vwind_impl_wgt=vwind_impl_wgt,
                    w=w[:, :n_lev],
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
            z_dwdz_dd,
            exner_dyn_incr,
            mass_flx_ic,
        )

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = random_field(grid, CEDim)
        mass_fl_e = random_field(grid, EdgeDim, KDim)
        z_theta_v_fl_e = random_field(grid, EdgeDim, KDim)
        z_flxdiv_mass = random_field(grid, CellDim, KDim)
        z_flxdiv_theta = random_field(grid, CellDim, KDim)
        z_w_expl = random_field(grid, CellDim, KDim, extend={KDim: 1})
        w_nnow = random_field(grid, CellDim, KDim)
        ddt_w_adv_ntl1 = random_field(grid, CellDim, KDim)
        ddt_w_adv_ntl2 = random_field(grid, CellDim, KDim)
        z_th_ddz_exner_c = random_field(grid, CellDim, KDim)
        z_contr_w_fl_l = random_field(grid, CellDim, KDim, extend={KDim: 1})
        rho_ic = random_field(grid, CellDim, KDim, extend={KDim: 1})
        w_concorr_c = random_field(grid, CellDim, KDim, extend={KDim: 1})
        vwind_expl_wgt = random_field(grid, CellDim)
        z_beta = random_field(grid, CellDim, KDim)
        exner_nnow = random_field(grid, CellDim, KDim)
        rho_nnow = random_field(grid, CellDim, KDim)
        theta_v_nnow = random_field(grid, CellDim, KDim)
        inv_ddqz_z_full = random_field(grid, CellDim, KDim)
        z_alpha = random_field(grid, CellDim, KDim, extend={KDim: 1})
        vwind_impl_wgt = random_field(grid, CellDim)
        theta_v_ic = random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_q = random_field(grid, CellDim, KDim)
        w = random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_rho_expl = random_field(grid, CellDim, KDim)
        z_exner_expl = random_field(grid, CellDim, KDim)
        exner_pr = random_field(grid, CellDim, KDim)
        ddt_exner_phy = random_field(grid, CellDim, KDim)
        rho_incr = random_field(grid, CellDim, KDim)
        exner_incr = random_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        z_raylfac = random_field(grid, KDim)
        exner_ref_mc = random_field(grid, CellDim, KDim)
        rho = random_field(grid, CellDim, KDim)
        exner = random_field(grid, CellDim, KDim)
        theta_v = random_field(grid, CellDim, KDim)
        z_dwdz_dd = random_field(grid, CellDim, KDim)
        exner_dyn_incr = random_field(grid, CellDim, KDim)
        mass_flx_ic = random_field(grid, CellDim, KDim)
        lprep_adv = True
        lclean_mflx = True
        r_nsubsteps = 0.5
        rd = 287.04
        cvd = 1004.64 - rd
        cvd_o_rd = cvd / rd
        rayleigh_klemp = 2
        idiv_method = 1
        l_open_ubc = False
        l_vert_nested = False
        is_iau_active = False
        rayleigh_type = 2
        divdamp_type = 3
        idyn_timestep = 1
        index_of_damping_layer = 9
        n_lev = 10
        jk_start = 0
        kstart_dd3d = 0
        kstart_moist = 1
        horizontal_lower_0 = 3316
        horizontal_upper_0 = 20896
        istep = 1
        dtime = 0.9
        wgt_nnow_vel = 0.25
        wgt_nnew_vel = 0.75
        cpd = 1004.64
        iau_wgt_dyn = 1.0
        lhdiff_rcf = True
        itime_scheme = 4
        horizontal_start = horizontal_lower_0 - 1
        horizontal_end = horizontal_upper_0 + 1
        vertical_start = 0
        vertical_end = n_lev
        vertical_lower = 0
        vertical_upper = n_lev + 1

        vert_idx = zero_field(grid, KDim, dtype=int32)
        for level in range(n_lev):
            vert_idx[level] = level

        horz_idx = zero_field(grid, CellDim, dtype=int32)
        for cell in range(grid.num_cells):
            horz_idx[cell] = cell

        return dict(
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
            z_dwdz_dd=z_dwdz_dd,
            exner_dyn_incr=exner_dyn_incr,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            mass_flx_ic=mass_flx_ic,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            itime_scheme=itime_scheme,
            lprep_adv=lprep_adv,
            lclean_mflx=lclean_mflx,
            r_nsubsteps=r_nsubsteps,
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
            istep=istep,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
