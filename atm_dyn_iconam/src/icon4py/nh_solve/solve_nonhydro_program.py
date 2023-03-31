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
from gt4py.next.common import Field
from gt4py.next.ffront.decorator import program
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_01 import (
    mo_solve_nonhydro_stencil_01,
)

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_02 import (
    mo_solve_nonhydro_stencil_02,
)

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_03 import (
    mo_solve_nonhydro_stencil_03,
)


from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)


from icon4py.atm_dyn_iconam.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)


from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
)


from icon4py.atm_dyn_iconam.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)

from icon4py.common.dimension import (
    C2E2CODim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    VertexDim,
)
from icon4py.state_utils.utils import _set_bool_c_k, _set_zero_c_k


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_2_3(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_02(
        exner_exfac, exner, exner_ref_mc, exner_pr, out=(z_exner_ex_pr, exner_pr)
    )
    _set_zero_c_k(self.z_exner_ex_pr, offset_provider={})


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_4_5_6(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_04(wgtfacq_c, z_exner_ex_pr, out=z_exner_ic)

    _mo_solve_nonhydro_stencil_05(wgtfac_c, z_exner_ex_pr, out=z_exner_ic)

    _mo_solve_nonhydro_stencil_06(z_exner_ic, inv_ddqz_z_full, out=z_dexner_dz_c_1)


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_7_8_9(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_07(
        rho, rho_ref_mc, theta_v, theta_ref_mc, out=(z_rth_pr_1, z_rth_pr_2)
    )
    _mo_solve_nonhydro_stencil_08(
        wgtfac_c,
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(rho_ic, z_rth_pr_1, z_rth_pr_2),
    )
    _mo_solve_nonhydro_stencil_09(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=(z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_11_lower_upper(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_11_lower(
        out=z_theta_v_pr_ic,
    )
    _mo_solve_nonhydro_stencil_11_upper(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
    )


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_43_44_45_45b(
    z_w_expl: Field[[CellDim, KDim], float],
    w_nnow: Field[[CellDim, KDim], float],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    z_beta: Field[[CellDim, KDim], float],
    exner_nnow: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    theta_v_nnow: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_q: Field[[CellDim, KDim], float],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
):
    _mo_solve_nonhydro_stencil_43(
        w_nnow,
        ddt_w_adv_ntl1,
        z_th_ddz_exner_c,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        dtime,
        cpd,
        out=(z_w_expl, z_contr_w_fl_l),
    )
    _mo_solve_nonhydro_stencil_44(
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
        out=(z_beta, z_alpha),
    )
    _mo_solve_nonhydro_stencil_45(out=z_alpha)
    _mo_solve_nonhydro_stencil_45_b(out=z_q)


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_47_48_49(
    w_nnew: Field[[CellDim, KDim], float],
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    rho_nnow: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_flxdiv_mass: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    z_flxdiv_theta: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    ddt_exner_phy: Field[[CellDim, KDim], float],
    dtime: float,
):
    _mo_solve_nonhydro_stencil_47(w_concorr_c, out=(w_nnew, z_contr_w_fl_l))
    _mo_solve_nonhydro_stencil_48(
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
        out=(z_rho_expl, z_exner_expl),
    )
    _mo_solve_nonhydro_stencil_49(
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
        out=(z_rho_expl, z_exner_expl),
    )


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_52_53(
    vwind_impl_wgt: Field[[CellDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    z_w_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    z_q: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    dtime: float,
    cpd: float,

):
    _mo_solve_nonhydro_stencil_52(
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_alpha,
        z_beta,
        z_w_expl,
        z_exner_expl,
        z_q,
        w,
        dtime,
        cpd,
        out=(z_q[:, 1:], w[:, 1:]),
    )
    _mo_solve_nonhydro_stencil_53_scan(z_q, w, out=w[:, 1:])


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_59_60(
    exner: Field[[CellDim, KDim], float],
    exner_dyn_incr: Field[[CellDim, KDim], float],
    ddt_exner_phy: Field[[CellDim, KDim], float],
    ndyn_substeps_var: float,
    dtime: float,
):
    _mo_solve_nonhydro_stencil_59(exner, out=exner_dyn_incr)
    _mo_solve_nonhydro_stencil_60(
        exner,
        ddt_exner_phy,
        exner_dyn_incr,
        ndyn_substeps_var,
        dtime,
        out=exner_dyn_incr,
    )


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_61_62(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    dtime: float,
):
    _mo_solve_nonhydro_stencil_61(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=(rho_new, exner_new, w_new),
    )
    _mo_solve_nonhydro_stencil_62(w_now, grf_tend_w, dtime, out=w_new)




@program(backend=gtfn_cpu.run_gtfn)
def predictor_tendencies(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    vt: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_w_concorr_mc: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    cell_endindex_local_minus1: int,
    edge_endindex_local_minus2: int,
    nflatlev_startindex: int,
    nlev: int,
):

    if LAM:
        _mo_solve_nonhydro_stencil_01()

    _mo_solve_nonhydro_stencil_02()
    _mo_solve_nonhydro_stencil_03()

    if(igradp_method <=3):
        _mo_solve_nonhydro_stencil_04()
        _mo_solve_nonhydro_stencil_05()
        _mo_solve_nonhydro_stencil_06()

        if (nflatlev == 1):

    _mo_solve_nonhydro_stencil_07()
    _mo_solve_nonhydro_stencil_08()
    _mo_solve_nonhydro_stencil_09()

    if (l_open_ubc and not l_vert_nested):

    _mo_solve_nonhydro_stencil_11_lower()
    _mo_solve_nonhydro_stencil_11_upper()

    if (igradp_method <= 3):
        _mo_solve_nonhydro_stencil_12()

    _mo_solve_nonhydro_stencil_13()


    if(iadv_rhotheta==1):
        mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
        mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl()
    elif(iadv_rhotheta==2):
        mo_math_gradients_grad_green_gauss_cell_dsl()
    elif (iadv_rhotheta == 3):
        upwind_hflux_miura3

    if (iadv_rhotheta <= 2):
        if(idiv_method == 1):

        else:

    _mo_solve_nonhydro_stencil_14()


    if(l_limited_area):
        _mo_solve_nonhydro_stencil_15()

    if (iadv_rhotheta == 2):
    else:
        mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1()

    _mo_solve_nonhydro_stencil_18()

    if (igradp_method <= 3):
        _mo_solve_nonhydro_stencil_19()
        _mo_solve_nonhydro_stencil_20()
    else if(igradp == 4 or igradp_method ==5):


    if (igradp_method == 3):
        _mo_solve_nonhydro_stencil_21()
    else if (igradp_method == 5):


    if (igradp_method == 3 or igradp_method==5):
        _mo_solve_nonhydro_stencil_22()

    _mo_solve_nonhydro_stencil_24()

    if (is_iau_active):
        _mo_solve_nonhydro_stencil_28()

    if (l_limited_area):
        _mo_solve_nonhydro_stencil_29()



     ##### COMMUNICATION PHASE

    _mo_solve_nonhydro_stencil_30()


    #####  Not sure about  _mo_solve_nonhydro_stencil_31()

    if (idiv_method == 1):
        _mo_solve_nonhydro_stencil_32()

    _mo_solve_nonhydro_stencil_35()

    _mo_solve_nonhydro_stencil_36()

    if(not l_vert_nested):
        _mo_solve_nonhydro_stencil_37()
        _mo_solve_nonhydro_stencil_38()

    _mo_solve_nonhydro_stencil_39()
    _mo_solve_nonhydro_stencil_40()

    if (idiv_method == 2):
        if (l_limited_area):
            init_zero_contiguous_dp()

        ##stencil not translated

    if (idiv_method == 2):
        div_avg()

    if (idiv_method == 1):
        _mo_solve_nonhydro_stencil_41()

    _mo_solve_nonhydro_stencil_43()

    _mo_solve_nonhydro_stencil_44()

    _mo_solve_nonhydro_stencil_45()
    _mo_solve_nonhydro_stencil_45_b()

    if (not l_open_ubc and not l_vert_nested):
        _mo_solve_nonhydro_stencil_46()

    _mo_solve_nonhydro_stencil_47()
    _mo_solve_nonhydro_stencil_48()
    _mo_solve_nonhydro_stencil_49()

    if (is_iau_active):
        _mo_solve_nonhydro_stencil_50()

    _mo_solve_nonhydro_stencil_52()
    _mo_solve_nonhydro_stencil_53()

    if(rayleigh_type == RAYLEIGH_KLEMP):
        ## ACC w_1 -> p_nh%w
        _mo_solve_nonhydro_stencil_54()

    _mo_solve_nonhydro_stencil_55()

    if(lhdiff_rcf and divdamp_type >=3):
        _mo_solve_nonhydro_stencil_56_63()

    if(idyn_timestep ==1):
        _mo_solve_nonhydro_stencil_59()
        _mo_solve_nonhydro_stencil_60()


    if(l_limited_area):  #for MPI-parallelized case
        _mo_solve_nonhydro_stencil_61()
        _mo_solve_nonhydro_stencil_62()

    if (lhdiff_rcf and divdamp_type >= 3):
        _mo_solve_nonhydro_stencil_56_63()

    ##### COMMUNICATION PHASE


    # Example. delete later
    _mo_velocity_advection_stencil_10(
        local_z_w_concorr_mc,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex + 1, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def advector_tendencies(
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    local_z_v_grad_w: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    local_z_ekinh: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    local_z_w_con_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    local_cfl_clipping: Field[[CellDim, KDim], bool],
    local_pre_levelmask: Field[[CellDim, KDim], bool],
    local_vcfl: Field[[CellDim, KDim], float],
    cfl_w_limit: float,
    dtime: float,
    z_w_con_c_full: Field[[CellDim, KDim], float],
    coeff1_dwdz: Field[[CellDim, KDim], float],
    coeff2_dwdz: Field[[CellDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    zeta: Field[[VertexDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    ddt_vn_adv: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    area: Field[[CellDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    owner_mask: Field[[CellDim], bool],
    levelmask: Field[[KDim], bool],
    scalfac_exdiff: float,
    area_edge: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    cell_startindex_nudging: int,
    cell_endindex_local_minus1: int,
    cell_endindex_local: int,
    edge_startindex_nudging_plus_one: int,
    edge_endindex_local: int,
    nflatlev_startindex: int,
    nlev: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_10()

    if (l_open_ubc and not l_vert_nested):


    _mo_solve_nonhydro_stencil_17()

    if(itime_scheme >=4):
        _mo_solve_nonhydro_stencil_23()


    if(lhdiff_rcf and (divdamp_order == 4 .OR. divdamp_order == 24)):
        _mo_solve_nonhydro_stencil_25()

    if (lhdiff_rcf):
        if(divdamp_order == 2 or (divdamp_order == 24 and scal_divdamp_o2 > 1e-6)):
            _mo_solve_nonhydro_stencil_26()
        if (divdamp_order == 4 or (divdamp_order == 24 and ivdamp_fac_o2 <= 4*divdamp_fac)):
            if(l_limited_area):
                _mo_solve_nonhydro_stencil_27()
            else:
                _mo_solve_nonhydro_4th_order_divdamp()


    if(is_iau_active):
        _mo_solve_nonhydro_stencil_28()

    ##### COMMUNICATION PHASE

    if (idiv_method == 1):
        _mo_solve_nonhydro_stencil_32()

        if(lpred_adv):
            if(lclean_mflx):
                _mo_solve_nonhydro_stencil_33()
            _mo_solve_nonhydro_stencil_34()

    if(itime_scheme >=5):
        _mo_solve_nonhydro_stencil_35()
        _mo_solve_nonhydro_stencil_39()
        _mo_solve_nonhydro_stencil_40()

    if (idiv_method == 2):
        if(l_limited_area):
            init_zero_contiguous_dp()

        ##stencil not translated

    if (idiv_method == 2):
        div_avg()

    if (idiv_method == 1):
        _mo_solve_nonhydro_stencil_41()

    if (itime_scheme>=4):
        _mo_solve_nonhydro_stencil_42()
    else:
        _mo_solve_nonhydro_stencil_43()

    _mo_solve_nonhydro_stencil_44()

    _mo_solve_nonhydro_stencil_45()
    _mo_solve_nonhydro_stencil_45_b()

    if(not l_open_ubc and not l_vert_nested):
        _mo_solve_nonhydro_stencil_46()

    _mo_solve_nonhydro_stencil_47()
    _mo_solve_nonhydro_stencil_48()
    _mo_solve_nonhydro_stencil_49()

    if(is_iau_active):
        _mo_solve_nonhydro_stencil_50()

    _mo_solve_nonhydro_stencil_52()
    _mo_solve_nonhydro_stencil_53()

    if (rayleigh_type == RAYLEIGH_KLEMP):
        ## ACC w_1 -> p_nh%w
        _mo_solve_nonhydro_stencil_54()

    _mo_solve_nonhydro_stencil_55()

    if (lpred_adv):
        if (lclean_mflx):
            _mo_solve_nonhydro_stencil_57()

    _mo_solve_nonhydro_stencil_58()

    if (lpred_adv):
        if (lclean_mflx):
            _mo_solve_nonhydro_stencil_64()
        _mo_solve_nonhydro_stencil_65()


    ##### COMMUNICATION PHASE




    _mo_velocity_advection_stencil_13(
        local_z_w_con_c,
        w_concorr_c,
        out=local_z_w_con_c,
        domain={
            CellDim: (4, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex + 1, nlev),
        },
    )

    _set_bool_c_k(out=local_cfl_clipping)
    _set_bool_c_k(out=local_pre_levelmask)
    _set_zero_c_k(out=local_vcfl)

