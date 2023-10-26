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
from gt4py.next import np_as_located_field
from gt4py.next.ffront.fbuiltins import int32
from test_mo_solve_nonhydro_stencil_41 import mo_solve_nonhydro_stencil_41_numpy
from test_mo_solve_nonhydro_stencil_42 import mo_solve_nonhydro_stencil_42_numpy
from test_mo_solve_nonhydro_stencil_43 import mo_solve_nonhydro_stencil_43_numpy
from test_mo_solve_nonhydro_stencil_44 import mo_solve_nonhydro_stencil_44_numpy
from test_mo_solve_nonhydro_stencil_45 import mo_solve_nonhydro_stencil_45_numpy
from test_mo_solve_nonhydro_stencil_46 import mo_solve_nonhydro_stencil_46_numpy
from test_mo_solve_nonhydro_stencil_47 import mo_solve_nonhydro_stencil_47_numpy
from test_mo_solve_nonhydro_stencil_49 import mo_solve_nonhydro_stencil_49_numpy
from test_mo_solve_nonhydro_stencil_50 import mo_solve_nonhydro_stencil_50_numpy
from test_mo_solve_nonhydro_stencil_52 import mo_solve_nonhydro_stencil_52_numpy
from test_mo_solve_nonhydro_stencil_53 import mo_solve_nonhydro_stencil_53_numpy
from test_mo_solve_nonhydro_stencil_54 import mo_solve_nonhydro_stencil_54_numpy
from test_mo_solve_nonhydro_stencil_55 import mo_solve_nonhydro_stencil_55_numpy
from test_mo_solve_nonhydro_stencil_58 import mo_solve_nonhydro_stencil_58_numpy

from icon4py.model.atmosphere.dycore.fused_stencils_41_to_60 import (
    fused_solve_nonhdyro_stencil_41_to_60,
)
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    flatten_first_two_dims,
    random_field,
    random_mask,
    zero_field,
)
from model.atmosphere.dycore.tests.stencil_tests.test_mo_solve_nonhydro_stencil_56_63 import (
    mo_solve_nonhydro_stencil_56_63_numpy,
)
from model.atmosphere.dycore.tests.stencil_tests.test_mo_solve_nonhydro_stencil_59 import (
    mo_solve_nonhydro_stencil_59_numpy,
)


class TestFusedMoSolveNonHydroStencil15To28(StencilTest):
    PROGRAM = fused_solve_nonhdyro_stencil_41_to_60
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
        mesh,
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
        k_field,
        w,
        z_rho_expl,
        z_exner_expl,
        exner_pr,
        ddt_exner_phy,
        rho_incr,
        exner_incr,
        ddqz_z_half,
        z_raylfac,
        w_1,
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
        horizontal_lower,
        horizontal_upper,
        istep,
    ):

        if istep == 1:
            if idiv_method == 1:
                # verified for e-9
                z_flxdiv_mass, z_flxdiv_theta = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    mo_solve_nonhydro_stencil_41_numpy(
                        mesh=mesh,
                        geofac_div=geofac_div,
                        mass_fl_e=mass_fl_e,
                        z_theta_v_fl_e=z_theta_v_fl_e,
                    ),
                    (z_flxdiv_mass, z_flxdiv_theta),
                )

            (z_w_expl, z_contr_w_fl_l) = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (k_field >= int32(1))
                & (k_field < n_lev),
                mo_solve_nonhydro_stencil_43_numpy(
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
            (z_beta, z_alpha) = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (k_field >= int32(0))
                & (k_field < n_lev),
                mo_solve_nonhydro_stencil_44_numpy(
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
            z_alpha = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (k_field == n_lev),
                mo_solve_nonhydro_stencil_45_numpy(mesh, z_alpha),
                z_alpha,
            )
            z_q = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (k_field == int32(0)),
                0,
                z_q,
            )

            if not (l_open_ubc and l_vert_nested):
                w, z_contr_w_fl_l = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx == 0),
                    mo_solve_nonhydro_stencil_46_numpy(
                        mesh=mesh,
                        w_nnew=w,
                        z_contr_w_fl_l=z_contr_w_fl_l,
                    ),
                    (w, z_contr_w_fl_l),
                )

            (w, z_contr_w_fl_l) = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (k_field == n_lev),
                mo_solve_nonhydro_stencil_47_numpy(w_concorr_c),
                (w, z_contr_w_fl_l),
            )
            # 48 and 49 are identical except for bounds
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (k_field >= int32(0))
                & (k_field < n_lev),
                mo_solve_nonhydro_stencil_49_numpy(
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

            if is_iau_active:
                (z_rho_expl, z_exner_expl) = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    mo_solve_nonhydro_stencil_50_numpy(
                        mesh=mesh,
                        z_rho_expl=z_rho_expl,
                        z_exner_expl=z_exner_expl,
                        rho_incr=rho_incr,
                        exner_incr=exner_incr,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    (z_rho_expl, z_exner_expl),
                )

            z_q, w = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (vert_idx >= 1),
                mo_solve_nonhydro_stencil_52_numpy(
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_ic,
                    ddqz_z_half=ddqz_z_half,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    z_w_expl=z_w_expl,
                    z_exner_expl=z_exner_expl,
                    z_q_ref=z_q,
                    w_ref=w,
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_q, w),
            )

            w = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (vert_idx >= 1),
                mo_solve_nonhydro_stencil_53_numpy(
                    z_q=z_q,
                    w=w,
                ),
                w,
            )

            if rayleigh_type == rayleigh_klemp:
                w = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx >= 1)
                    & (vert_idx < (index_of_damping_layer + 1)),
                    mo_solve_nonhydro_stencil_54_numpy(
                        mesh=mesh,
                        z_raylfac=z_raylfac,
                        w_1=w_1,
                        w=w,
                    ),
                    w,
                )

            rho, exner, theta_v = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx >= jk_start),
                mo_solve_nonhydro_stencil_55_numpy(
                    mesh=mesh,
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
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx >= kstart_dd3d),
                    mo_solve_nonhydro_stencil_56_63_numpy(
                        inv_ddqz_z_full=inv_ddqz_z_full, w=w, w_concorr_c=w_concorr_c
                    ),
                    z_dwdz_dd,
                )

            if idyn_timestep == 1:
                exner_dyn_incr = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx >= kstart_moist),
                    mo_solve_nonhydro_stencil_59_numpy(exner=exner),
                    exner_dyn_incr,
                )

        else:

            if idiv_method == 1:
                # verified for e-9
                z_flxdiv_mass, z_flxdiv_theta = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    mo_solve_nonhydro_stencil_41_numpy(
                        mesh=mesh,
                        geofac_div=geofac_div,
                        mass_fl_e=mass_fl_e,
                        z_theta_v_fl_e=z_theta_v_fl_e,
                    ),
                    (z_flxdiv_mass, z_flxdiv_theta),
                )

            if itime_scheme == 4:

                (z_w_expl, z_contr_w_fl_l) = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (k_field >= int32(1))
                    & (k_field < n_lev),
                    mo_solve_nonhydro_stencil_42_numpy(
                        mesh,
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

                (z_beta, z_alpha) = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (k_field >= int32(0))
                    & (k_field < n_lev),
                    mo_solve_nonhydro_stencil_44_numpy(
                        mesh,
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
                z_alpha = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (k_field == n_lev),
                    mo_solve_nonhydro_stencil_45_numpy(mesh, z_alpha),
                    z_alpha,
                )

                z_q = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (k_field == 0),
                    0,
                    z_q,
                )

            else:
                (z_w_expl, z_contr_w_fl_l) = np.where(
                    (k_field >= int32(1)) & (k_field < n_lev),
                    mo_solve_nonhydro_stencil_43_numpy(
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
                (z_beta, z_alpha) = np.where(
                    (k_field >= int32(0)) & (k_field < n_lev),
                    mo_solve_nonhydro_stencil_44_numpy(
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
                z_alpha = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (k_field == n_lev),
                    mo_solve_nonhydro_stencil_45_numpy(mesh, z_alpha),
                    z_alpha,
                )
                z_q = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (k_field == int32(0)),
                    0,
                    z_q,
                )

            if not (l_open_ubc and not l_vert_nested):
                w, z_contr_w_fl_l = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx == 0),
                    mo_solve_nonhydro_stencil_46_numpy(
                        mesh=mesh,
                        w_nnew=w,
                        z_contr_w_fl_l=z_contr_w_fl_l,
                    ),
                    (w, z_contr_w_fl_l),
                )

            (w, z_contr_w_fl_l) = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (k_field == n_lev),
                mo_solve_nonhydro_stencil_47_numpy(w_concorr_c),
                (w, z_contr_w_fl_l),
            )
            # 48 and 49 are identical except for bounds
            (z_rho_expl, z_exner_expl) = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (k_field >= int32(0))
                & (k_field < n_lev),
                mo_solve_nonhydro_stencil_49_numpy(
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

            if is_iau_active:
                (z_rho_expl, z_exner_expl) = np.where(
                    (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                    mo_solve_nonhydro_stencil_50_numpy(
                        mesh=mesh,
                        z_rho_expl=z_rho_expl,
                        z_exner_expl=z_exner_expl,
                        rho_incr=rho_incr,
                        exner_incr=exner_incr,
                        iau_wgt_dyn=iau_wgt_dyn,
                    ),
                    (z_rho_expl, z_exner_expl),
                )

            z_q, w = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (vert_idx >= 1),
                mo_solve_nonhydro_stencil_52_numpy(
                    vwind_impl_wgt=vwind_impl_wgt,
                    theta_v_ic=theta_v_ic,
                    ddqz_z_half=ddqz_z_half,
                    z_alpha=z_alpha,
                    z_beta=z_beta,
                    z_w_expl=z_w_expl,
                    z_exner_expl=z_exner_expl,
                    z_q_ref=z_q,
                    w_ref=w,
                    dtime=dtime,
                    cpd=cpd,
                ),
                (z_q, w),
            )

            w = np.where(
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper) & (vert_idx >= 1),
                mo_solve_nonhydro_stencil_53_numpy(
                    z_q=z_q,
                    w=w,
                ),
                w,
            )

            if rayleigh_type == rayleigh_klemp:
                w = np.where(
                    (horizontal_lower <= horz_idx)
                    & (horz_idx < horizontal_upper)
                    & (vert_idx >= 1)
                    & (vert_idx < (index_of_damping_layer + 1)),
                    mo_solve_nonhydro_stencil_54_numpy(
                        mesh=mesh,
                        z_raylfac=z_raylfac,
                        w_1=w_1,
                        w=w,
                    ),
                    w,
                )

            rho, exner, theta_v = np.where(
                (horizontal_lower <= horz_idx)
                & (horz_idx < horizontal_upper)
                & (vert_idx >= jk_start),
                mo_solve_nonhydro_stencil_55_numpy(
                    mesh=mesh,
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
                (horizontal_lower <= horz_idx) & (horz_idx < horizontal_upper),
                mo_solve_nonhydro_stencil_58_numpy(
                    mesh=mesh,
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
            z_dwdz_dd,
            exner_dyn_incr,
            mass_flx_ic,
        )
