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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32
from .test_add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation_numpy,
)
from .test_apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism_numpy,
)
from .test_compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping_numpy,
)
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
from .test_compute_solver_coefficients_matrix import (
    compute_solver_coefficients_matrix_numpy,
)
from .test_set_lower_boundary_condition_for_w_and_contravariant_correction import (
    set_lower_boundary_condition_for_w_and_contravariant_correction_numpy,
)
from .test_solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution_numpy,
)
from .test_solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep_numpy,
)
from .test_update_mass_volume_flux import (
    update_mass_volume_flux_numpy,
)

from icon4py.model.atmosphere.dycore.fused_solve_nonhydro_stencil_41_to_60 import (
    fused_solve_nonhydro_stencil_41_to_60,
)
from icon4py.model.atmosphere.dycore.solve_nonhydro import TimeSteppingScheme
from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers


def compute_divergence_of_fluxes_of_rho_and_theta_numpy(
    grid,
    geofac_div: np.ndarray,
    mass_fl_e: np.ndarray,
    z_theta_v_fl_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e = grid[dims.C2EDim]
    c2ce = helpers.as_1d_connectivity(c2e)
    geofac_div = np.expand_dims(geofac_div, axis=-1)

    z_flxdiv_mass_wp = np.sum(geofac_div[c2ce] * mass_fl_e[c2e], axis=1)
    z_flxdiv_theta_wp = np.sum(geofac_div[c2ce] * z_theta_v_fl_e[c2e], axis=1)
    return (z_flxdiv_mass_wp, z_flxdiv_theta_wp)


class TestFusedMoSolveNonHydroStencil41To60(helpers.StencilTest):
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
        "vol_flx_ic",
    )

    # flake8: noqa: C901
    @classmethod
    def reference(
        cls,
        grid,
        geofac_div: np.ndarray,
        mass_fl_e: np.ndarray,
        z_theta_v_fl_e: np.ndarray,
        z_flxdiv_mass: np.ndarray,
        z_flxdiv_theta: np.ndarray,
        z_w_expl: np.ndarray,
        w_nnow: np.ndarray,
        ddt_w_adv_ntl1: np.ndarray,
        ddt_w_adv_ntl2: np.ndarray,
        z_th_ddz_exner_c: np.ndarray,
        z_contr_w_fl_l: np.ndarray,
        rho_ic: np.ndarray,
        w_concorr_c: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        z_beta: np.ndarray,
        exner_nnow: np.ndarray,
        rho_nnow: np.ndarray,
        theta_v_nnow: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        z_alpha: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        theta_v_ic: np.ndarray,
        z_q: np.ndarray,
        w: np.ndarray,
        z_rho_expl: np.ndarray,
        z_exner_expl: np.ndarray,
        exner_pr: np.ndarray,
        ddt_exner_phy: np.ndarray,
        rho_incr: np.ndarray,
        exner_incr: np.ndarray,
        ddqz_z_half: np.ndarray,
        z_raylfac: np.ndarray,
        exner_ref_mc: np.ndarray,
        rho: np.ndarray,
        exner: np.ndarray,
        theta_v: np.ndarray,
        z_dwdz_dd: np.ndarray,
        exner_dyn_incr: np.ndarray,
        vert_idx: np.ndarray,
        mass_flx_ic: np.ndarray,
        vol_flx_ic: np.ndarray,
        wgt_nnow_vel: float,
        wgt_nnew_vel: float,
        itime_scheme: float,
        lprep_adv: bool,
        r_nsubsteps: float,
        cvd_o_rd: float,
        iau_wgt_dyn: float,
        dtime: float,
        rd: float,
        cvd: float,
        cpd: float,
        rayleigh_klemp: int,
        l_vert_nested: bool,
        is_iau_active: bool,
        rayleigh_type: int,
        divdamp_type: int,
        index_of_damping_layer: int,
        n_lev: int,
        jk_start: int,
        kstart_dd3d: int,
        kstart_moist: int,
        start_cell_nudging: int,
        end_cell_local: int,
        istep: int,
        at_first_substep: bool,
        vertical_start: int,
        vertical_end: int,
    ):
        horz_idx = np.asarray(np.arange(vol_flx_ic.shape[0]))
        horz_idx = horz_idx[:, np.newaxis]
        if istep == 1:
            # verified for e-9
            z_flxdiv_mass, z_flxdiv_theta = np.where(
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local),
                compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                    grid=grid,
                    geofac_div=geofac_div,
                    mass_fl_e=mass_fl_e,
                    z_theta_v_fl_e=z_theta_v_fl_e,
                ),
                (z_flxdiv_mass, z_flxdiv_theta),
            )

            (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
            z_alpha[:, :n_lev] = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
                & (vert_idx == n_lev),
                0.0,  # _init_cell_kdim_field_with_zero_vp_numpy(grid=grid, z_alpha=z_alpha[:, :n_lev]),
                z_alpha[:, :n_lev],
            )
            z_q = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
                & (vert_idx == int32(0)),
                0.0,
                z_q,
            )

            if not l_vert_nested:
                w[start_cell_nudging:end_cell_local, :n_lev] = 0.0
                z_contr_w_fl_l[start_cell_nudging:end_cell_local, :n_lev] = 0.0

            (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
                    (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local),
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
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local) & (vert_idx >= 1),
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
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local) & (vert_idx >= 1),
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
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
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
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
            if divdamp_type >= 3:
                z_dwdz_dd = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx >= kstart_dd3d),
                    compute_dwdz_for_divergence_damping_numpy(
                        grid=grid, inv_ddqz_z_full=inv_ddqz_z_full, w=w, w_concorr_c=w_concorr_c
                    ),
                    z_dwdz_dd,
                )

            if at_first_substep:
                exner_dyn_incr = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx >= kstart_moist),
                    exner_dyn_incr.astype(fa.vpfloat),
                    # _copy_cell_kdim_field_to_vp_numpy(),
                    exner_dyn_incr,
                )

        else:
            # verified for e-9
            z_flxdiv_mass, z_flxdiv_theta = np.where(
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local),
                compute_divergence_of_fluxes_of_rho_and_theta_numpy(
                    grid=grid,
                    geofac_div=geofac_div,
                    mass_fl_e=mass_fl_e,
                    z_theta_v_fl_e=z_theta_v_fl_e,
                ),
                (z_flxdiv_mass, z_flxdiv_theta),
            )

            if itime_scheme == TimeSteppingScheme.MOST_EFFICIENT:
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
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
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
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
                z_alpha[:, :n_lev] = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx == n_lev),
                    0.0,  # _init_cell_kdim_field_with_zero_vp_numpy(grid=grid, z_alpha=z_alpha[:, :n_lev]),
                    z_alpha[:, :n_lev],
                )

                z_q = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx == 0),
                    0.0,
                    z_q,
                )

            else:
                (z_w_expl[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
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
                (z_beta, z_alpha) = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx >= int32(0))
                    & (vert_idx < n_lev),
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
                z_alpha = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx == n_lev),
                    0.0,  # _init_cell_kdim_field_with_zero_vp_numpy(grid=grid, z_alpha=z_alpha),
                    z_alpha,
                )
                z_q = np.where(
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
                    & (vert_idx == int32(0)),
                    0.0,
                    z_q,
                )

            if not l_vert_nested:
                w[start_cell_nudging:end_cell_local, :n_lev] = 0.0
                z_contr_w_fl_l[start_cell_nudging:end_cell_local, :n_lev] = 0.0
                # w[:, :n_lev], z_contr_w_fl_l[:, :n_lev] = np.where(
                #     (start_cell_nudging <= horz_idx)
                #     & (horz_idx < end_cell_local)
                #     & (vert_idx == 0),
                #     (0., 0.),
                #     # _init_two_cell_kdim_fields_with_zero_wp_numpy(
                #     #     grid=grid,
                #     #     w_nnew=w[:, :n_lev],
                #     #     z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                #     # ),
                #     (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
                # )

            (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]) = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
                & (vert_idx == n_lev),
                set_lower_boundary_condition_for_w_and_contravariant_correction_numpy(
                    grid, w_concorr_c[:, :n_lev], z_contr_w_fl_l[:, :n_lev]
                ),
                (w[:, :n_lev], z_contr_w_fl_l[:, :n_lev]),
            )
            # 48 and 49 are identical except for bounds
            (z_rho_expl, z_exner_expl) = np.where(
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
                    (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local),
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
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local) & (vert_idx >= 1),
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
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local) & (vert_idx >= 1),
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
                    (start_cell_nudging <= horz_idx)
                    & (horz_idx < end_cell_local)
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
                (start_cell_nudging <= horz_idx)
                & (horz_idx < end_cell_local)
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
                if at_first_substep:
                    mass_flx_ic = np.zeros_like(exner)
                    vol_flx_ic = np.zeros_like(exner)

            (mass_flx_ic, vol_flx_ic) = np.where(
                (start_cell_nudging <= horz_idx) & (horz_idx < end_cell_local),
                update_mass_volume_flux_numpy(
                    grid=grid,
                    z_contr_w_fl_l=z_contr_w_fl_l[:, :n_lev],
                    rho_ic=rho_ic[:, :n_lev],
                    vwind_impl_wgt=vwind_impl_wgt,
                    w=w[:, :n_lev],
                    mass_flx_ic=mass_flx_ic,
                    r_nsubsteps=r_nsubsteps,
                ),
                (mass_flx_ic, vol_flx_ic),
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
            vol_flx_ic,
        )

    @pytest.fixture
    def input_data(self, grid):
        geofac_div = data_alloc.random_field(grid, dims.CEDim)
        mass_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_theta_v_fl_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        z_flxdiv_mass = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_flxdiv_theta = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_w_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        w_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddt_w_adv_ntl1 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddt_w_adv_ntl2 = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_th_ddz_exner_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_contr_w_fl_l = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        rho_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        w_concorr_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        vwind_expl_wgt = data_alloc.random_field(grid, dims.CellDim)
        z_beta = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        theta_v_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_alpha = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        vwind_impl_wgt = data_alloc.random_field(grid, dims.CellDim)
        theta_v_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_q = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_rho_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_exner_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddt_exner_phy = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho_incr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_incr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_raylfac = data_alloc.random_field(grid, dims.KDim)
        exner_ref_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_dwdz_dd = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        exner_dyn_incr = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        mass_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        vol_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        lprep_adv = True
        r_nsubsteps = 0.5
        rayleigh_klemp = 2
        l_vert_nested = False
        is_iau_active = False
        at_first_substep = True
        rayleigh_type = 2
        divdamp_type = 3
        index_of_damping_layer = 9
        jk_start = 0
        kstart_dd3d = 0
        kstart_moist = 1
        istep = 1
        dtime = 0.9
        veladv_offctr = 0.25
        wgt_nnow_vel = 0.5 - veladv_offctr
        wgt_nnew_vel = 0.5 + veladv_offctr
        iau_wgt_dyn = 1.0
        itime_scheme = 4

        cell_domain = h_grid.domain(dims.CellDim)
        start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        vert_idx = data_alloc.index_field(grid=grid, dim=dims.KDim)

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
            vert_idx=vert_idx,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            itime_scheme=itime_scheme,
            lprep_adv=lprep_adv,
            r_nsubsteps=r_nsubsteps,
            cvd_o_rd=(constants.CVD / constants.RD),
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
            rd=constants.RD,
            cvd=constants.CVD,
            cpd=constants.CPD,
            rayleigh_klemp=rayleigh_klemp,
            l_vert_nested=l_vert_nested,
            is_iau_active=is_iau_active,
            rayleigh_type=rayleigh_type,
            divdamp_type=divdamp_type,
            index_of_damping_layer=index_of_damping_layer,
            n_lev=grid.num_levels,
            jk_start=jk_start,
            kstart_dd3d=kstart_dd3d,
            kstart_moist=kstart_moist,
            start_cell_nudging=start_cell_nudging,
            end_cell_local=end_cell_local,
            istep=istep,
            at_first_substep=at_first_substep,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
