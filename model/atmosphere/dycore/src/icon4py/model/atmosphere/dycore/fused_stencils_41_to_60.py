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
from gt4py.next.ffront.fbuiltins import Field, int32, where

import icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program as nhsolve_prog
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_41 import (
    _mo_solve_nonhydro_stencil_41,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_46 import (
    _mo_solve_nonhydro_stencil_46,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_50 import (
    _mo_solve_nonhydro_stencil_50,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_52 import (
    _mo_solve_nonhydro_stencil_52,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_53 import (
    _mo_solve_nonhydro_stencil_53_scan,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_54 import (
    _mo_solve_nonhydro_stencil_54,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_55 import (
    _mo_solve_nonhydro_stencil_55,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_56_63 import (
    _mo_solve_nonhydro_stencil_56_63,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_58 import (
    _mo_solve_nonhydro_stencil_58,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_59 import (
    _mo_solve_nonhydro_stencil_59,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _set_zero_c_k
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_solve_nonhydro_stencil_41_to_60_predictor(
    geofac_div: Field[[CEDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    z_flxdiv_mass: Field[[CellDim, KDim], float],
    z_flxdiv_theta: Field[[CellDim, KDim], float],
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
    k_field: Field[[KDim], int32],
    w: Field[[CellDim, KDim], float],
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    ddt_exner_phy: Field[[CellDim, KDim], float],
    rho_incr: Field[[CellDim, KDim], float],
    exner_incr: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_raylfac: Field[[KDim], float],
    w_1: Field[[CellDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    exner_dyn_incr: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    cvd_o_rd: float,
    iau_wgt_dyn: float,
    dtime: float,
    rd: float,
    cvd: float,
    cpd: float,
    rayleigh_klemp: float,
    idiv_method: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    rayleigh_type: float,
    lhdiff_rcf: bool,
    divdamp_type: int32,
    idyn_timestep: int32,
    index_of_damping_layer: int32,
    n_lev: int32,
    jk_start: int32,
    kstart_dd3d: int32,
    kstart_moist: int32,
    horizontal_lower: int32,
    horizontal_upper: int32,
    horizontal_lower_1: int32,
    horizontal_upper_1: int32,
):
    # horizontal_lower = start_cell_nudging
    # horizontal_upper = end_cell_local
    # horizontal_lower_1 = cell_startindex_nudging_plus1
    # horizontal_upper_1 = cell_endindex_interior

    if idiv_method == 1:
        z_flxdiv_mass, z_flxdiv_theta = where(
            horizontal_lower <= horz_idx < horizontal_upper,
            _mo_solve_nonhydro_stencil_41(
                geofac_div=geofac_div,
                mass_fl_e=mass_fl_e,
                z_theta_v_fl_e=z_theta_v_fl_e,
            ),
            (z_flxdiv_mass, z_flxdiv_theta),
        )

    z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < (n_lev + int32(1))),
        nhsolve_prog._stencils_43_44_45_45b(
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
            k_field=k_field,
            rd=rd,
            cvd=cvd,
            dtime=dtime,
            cpd=cpd,
            nlev=n_lev,
        ),
        (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
    )

    if not (l_open_ubc and l_vert_nested):
        w, z_contr_w_fl_l = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < int32(1)),
            _mo_solve_nonhydro_stencil_46(),
            (w, z_contr_w_fl_l),
        )

    w_nnew, z_contr_w_fl_l, z_rho_expl, z_exner_expl = where(
        (horizontal_lower <= horz_idx < horizontal_upper)
        & (n_lev <= vert_idx < (n_lev + int32(1))),
        nhsolve_prog._stencils_47_48_49(
            w_nnew=w,
            z_contr_w_fl_l=z_contr_w_fl_l,
            w_concorr_c=w_concorr_c,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_nnow=rho_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_flxdiv_mass=z_flxdiv_mass,
            exner_pr=exner_pr,
            z_beta=z_beta,
            z_flxdiv_theta=z_flxdiv_theta,
            theta_v_ic=theta_v_ic,
            ddt_exner_phy=ddt_exner_phy,
            k_field=k_field,
            dtime=dtime,
            cell_startindex_nudging_plus1=horizontal_lower_1,
            cell_endindex_interior=horizontal_upper_1,
            nlev=n_lev,
            nlev_k=n_lev + 1,
        ),
        (w, z_contr_w_fl_l, z_rho_expl, z_exner_expl),
    )

    if is_iau_active:
        z_rho_expl, z_exner_expl = where(
            horizontal_lower <= horz_idx < horizontal_upper,
            _mo_solve_nonhydro_stencil_50(
                z_rho_expl=z_rho_expl,
                z_exner_expl=z_exner_expl,
                rho_incr=rho_incr,
                exner_incr=exner_incr,
                iau_wgt_dyn=iau_wgt_dyn,
            ),
            (z_rho_expl, z_exner_expl),
        )

    z_q, w = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(1) <= vert_idx),
        _mo_solve_nonhydro_stencil_52(
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
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(1) <= vert_idx),
        _mo_solve_nonhydro_stencil_53_scan(
            z_q=z_q,
            w=w,
        ),
        w,
    )

    if rayleigh_type == rayleigh_klemp:
        w = where(
            (horizontal_lower <= horz_idx < horizontal_upper)
            & (int32(1) <= vert_idx < (index_of_damping_layer + int32(1))),
            _mo_solve_nonhydro_stencil_54(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=w,
            ),
            w,
        )

    rho, exner, theta_v = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(jk_start) <= vert_idx),
        _mo_solve_nonhydro_stencil_55(
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
    if lhdiff_rcf and divdamp_type >= 3:
        z_dwdz_dd = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (kstart_dd3d <= vert_idx),
            _mo_solve_nonhydro_stencil_56_63(
                inv_ddqz_z_full=inv_ddqz_z_full,
                w=w,
                w_concorr_c=w_concorr_c,
            ),
            z_dwdz_dd,
        )

    if idyn_timestep == 1:
        exner_dyn_incr = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (kstart_moist <= vert_idx),
            _mo_solve_nonhydro_stencil_59(exner=exner_nnow),
            exner_dyn_incr,
        )


@field_operator
def fused_solve_nonhdyro_stencil_41_to_60_corrector(
    geofac_div: Field[[CEDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    z_flxdiv_mass: Field[[CellDim, KDim], float],
    z_flxdiv_theta: Field[[CellDim, KDim], float],
    z_w_expl: Field[[CellDim, KDim], float],
    w_nnow: Field[[CellDim, KDim], float],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], float],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], float],
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
    k_field: Field[[KDim], int32],
    w: Field[[CellDim, KDim], float],
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    ddt_exner_phy: Field[[CellDim, KDim], float],
    rho_incr: Field[[CellDim, KDim], float],
    exner_incr: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_raylfac: Field[[KDim], float],
    w_1: Field[[CellDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    rho: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    horz_idx: Field[[CellDim], int32],
    vert_idx: Field[[KDim], int32],
    cvd_o_rd: float,
    iau_wgt_dyn: float,
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: int32,
    idiv_method: int32,
    itime_scheme: int32,
    l_open_ubc: bool,
    l_vert_nested: bool,
    is_iau_active: bool,
    index_of_damping_layer: int32,
    rayleigh_klemp: float,
    rayleigh_type: float,
    jk_start: int32,
    lprep_adv: float,
    lclean_mflx: float,
    r_nsubsteps: float,
    horizontal_lower: int32,
    horizontal_upper: int32,
    horizontal_lower_1: int32,
    horizontal_upper_1: int32,
):
    # horizontal_lower = start_cell_nudging
    # horizontal_upper = end_cell_local
    # horizontal_lower_1 = cell_startindex_nudging_plus1
    # horizontal_upper_1 = cell_endindex_interior

    if idiv_method == 1:
        # verified for e-9
        z_flxdiv_mass, z_flxdiv_theta = where(
            horizontal_lower <= horz_idx < horizontal_upper,
            _mo_solve_nonhydro_stencil_41(
                geofac_div=geofac_div,
                mass_fl_e=mass_fl_e,
                z_theta_v_fl_e=z_theta_v_fl_e,
            ),
            (z_flxdiv_mass, z_flxdiv_theta),
        )

    if itime_scheme == 4:
        (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q) = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < nlev + 1),
            nhsolve_prog._stencils_42_44_45_45b(
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
                k_field=k_field,
                rd=rd,
                cvd=cvd,
                dtime=dtime,
                cpd=cpd,
                wgt_nnow_vel=wgt_nnow_vel,
                wgt_nnew_vel=wgt_nnew_vel,
                nlev=nlev,
            ),
            (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
        )
    else:
        (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q) = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < nlev + 1),
            nhsolve_prog._stencils_43_44_45_45b(
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
                k_field=k_field,
                rd=rd,
                cvd=cvd,
                dtime=dtime,
                cpd=cpd,
                nlev=nlev,
            ),
            (z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
        )

    if not l_open_ubc and not l_vert_nested:
        w, z_contr_w_fl_l = where(
            (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < int32(1)),
            _mo_solve_nonhydro_stencil_46(),
            (w, z_contr_w_fl_l),
        )

    w, z_contr_w_fl_l, z_rho_expl, z_exner_expl = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (nlev <= vert_idx < (nlev + int32(1))),
        nhsolve_prog._stencils_47_48_49(
            w_nnew=w,
            z_contr_w_fl_l=z_contr_w_fl_l,
            w_concorr_c=w_concorr_c,
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_nnow=rho_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_flxdiv_mass=z_flxdiv_mass,
            exner_pr=exner_pr,
            z_beta=z_beta,
            z_flxdiv_theta=z_flxdiv_theta,
            theta_v_ic=theta_v_ic,
            ddt_exner_phy=ddt_exner_phy,
            k_field=k_field,
            dtime=dtime,
            cell_startindex_nudging_plus1=horizontal_lower_1,
            cell_endindex_interior=horizontal_upper_1,
            nlev=nlev,
            nlev_k=nlev + 1,
        ),
        (w, z_contr_w_fl_l, z_rho_expl, z_exner_expl),
    )

    if is_iau_active:
        z_rho_expl, z_exner_expl = where(
            horizontal_lower <= horz_idx < horizontal_upper,
            _mo_solve_nonhydro_stencil_50(
                z_rho_expl=z_rho_expl,
                z_exner_expl=z_exner_expl,
                rho_incr=rho_incr,
                exner_incr=exner_incr,
                iau_wgt_dyn=iau_wgt_dyn,
            ),
            (z_rho_expl, z_exner_expl),
        )

    z_q, w = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(1) <= vert_idx),
        _mo_solve_nonhydro_stencil_52(
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
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(1) <= vert_idx),
        _mo_solve_nonhydro_stencil_53_scan(
            z_q=z_q,
            w=w,
        ),
        w,
    )

    if rayleigh_type == rayleigh_klemp:
        w = where(
            (horizontal_lower <= horz_idx < horizontal_upper)
            & (int32(1) <= vert_idx < (index_of_damping_layer + int32(1))),
            _mo_solve_nonhydro_stencil_54(
                z_raylfac=z_raylfac,
                w_1=w_1,
                w=w,
            ),
            w,
        )

    rho, exner, theta_v = where(
        (horizontal_lower <= horz_idx < horizontal_upper) & (int32(jk_start) <= vert_idx),
        _mo_solve_nonhydro_stencil_55(
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

    if lprep_adv:
        if lclean_mflx:
            mass_flx_ic = where(
                (horizontal_lower <= horz_idx < horizontal_upper),
                _set_zero_c_k(),
                mass_flx_ic,
            )

    mass_flx_ic = where(
        (horizontal_lower <= horz_idx < horizontal_upper),
        _mo_solve_nonhydro_stencil_58(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            r_nsubsteps=r_nsubsteps,
        ),
        mass_flx_ic,
    )
