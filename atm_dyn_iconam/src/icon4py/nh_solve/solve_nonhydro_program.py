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
from gt4py.next.ffront.fbuiltins import maximum
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_02 import (
    _mo_solve_nonhydro_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_04 import (
    _mo_solve_nonhydro_stencil_04,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_05 import (
    _mo_solve_nonhydro_stencil_05,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_06 import (
    _mo_solve_nonhydro_stencil_06,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_07 import (
    _mo_solve_nonhydro_stencil_07,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_08 import (
    _mo_solve_nonhydro_stencil_08,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_09 import (
    _mo_solve_nonhydro_stencil_09,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_11_lower import (
    _mo_solve_nonhydro_stencil_11_lower,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_11_upper import (
    _mo_solve_nonhydro_stencil_11_upper,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_35 import (
    _mo_solve_nonhydro_stencil_35,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_36 import (
    _mo_solve_nonhydro_stencil_36,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_37 import (
    _mo_solve_nonhydro_stencil_37,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_38 import (
    _mo_solve_nonhydro_stencil_38,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_39 import (
    _mo_solve_nonhydro_stencil_39,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_40 import (
    _mo_solve_nonhydro_stencil_40,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_43 import (
    _mo_solve_nonhydro_stencil_43,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_44 import (
    _mo_solve_nonhydro_stencil_44,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_45 import (
    _mo_solve_nonhydro_stencil_45,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_45_b import (
    _mo_solve_nonhydro_stencil_45_b,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_47 import (
    _mo_solve_nonhydro_stencil_47,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_48 import (
    _mo_solve_nonhydro_stencil_48,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_49 import (
    _mo_solve_nonhydro_stencil_49,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_52 import (
    _mo_solve_nonhydro_stencil_52,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_53 import (
    _mo_solve_nonhydro_stencil_53_scan,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_56_63 import (
    _mo_solve_nonhydro_stencil_56_63,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_59 import (
    _mo_solve_nonhydro_stencil_59,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_60 import (
    _mo_solve_nonhydro_stencil_60,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_61 import (
    _mo_solve_nonhydro_stencil_61,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_62 import (
    _mo_solve_nonhydro_stencil_62,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_66 import (
    _mo_solve_nonhydro_stencil_66,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_67 import (
    _mo_solve_nonhydro_stencil_67,
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
    V2CDim,
    VertexDim,
)
from icon4py.state_utils.utils import _set_zero_c_k


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_2_3(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    cell_endindex_interior_minus1: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_02(
        exner_exfac,
        exner,
        exner_ref_mc,
        exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (0, nlev)},
    )
    _set_zero_c_k(
        z_exner_ex_pr,
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (0, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_4_5_6(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    cell_endindex_interior_minus1: int,
    nflatlev_startindex: int,
    nlev: int,
    nlevp1: int,
):
    # Perturbation Exner pressure on bottom half level
    _mo_solve_nonhydro_stencil_04(
        wgtfacq_c,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (nlevp1, nlevp1)},
    )

    # WS: moved full z_exner_ic calculation here to avoid OpenACC dependency on jk+1 below
    # possibly GZ will want to consider the cache ramifications of this change for CPU
    _mo_solve_nonhydro_stencil_05(
        wgtfac_c,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={
            CellDim: (2, cell_endindex_interior_minus1),
            KDim: (maximum(1, nflatlev_startindex), nlev),
        },
    )

    # First vertical derivative of perturbation Exner pressure
    _mo_solve_nonhydro_stencil_06(
        z_exner_ic,
        inv_ddqz_z_full,
        out=z_dexner_dz_c_1,
        domain={
            CellDim: (2, cell_endindex_interior_minus1),
            KDim: (maximum(1, nflatlev_startindex), nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_7_8_9(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
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
    cell_endindex_interior_minus1: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_07(
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(z_rth_pr_1, z_rth_pr_2),
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (0, 0)},
    )
    _mo_solve_nonhydro_stencil_08(
        wgtfac_c,
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(rho_ic, z_rth_pr_1, z_rth_pr_2),
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (1, nlev)},
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
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (1, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_11_lower_upper(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    cell_endindex_interior_minus1: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_11_lower(
        out=z_theta_v_pr_ic,
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (0, 0)},
    )
    _mo_solve_nonhydro_stencil_11_upper(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={CellDim: (2, cell_endindex_interior_minus1), KDim: (nlevp1, nlevp1)},
    )


# TODO: @nfarabullini: what's up with stencil_20?


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_35_36(
    vn: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    edge_endindex_interior_minus2: int,
    nflatlev_startindex: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_35(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain={
            EdgeDim: (4, edge_endindex_interior_minus2),
            KDim: (nflatlev_startindex, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_36(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={EdgeDim: (4, edge_endindex_interior_minus2), KDim: (1, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_37_38(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    wgtfacq_e: Field[[EdgeDim, KDim], float],
    edge_endindex_local_minus2: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_37(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={EdgeDim: (4, edge_endindex_local_minus2), KDim: (0, 0)},
    )
    _mo_solve_nonhydro_stencil_38(
        vn,
        wgtfacq_e,
        out=vn_ie,
        domain={EdgeDim: (4, edge_endindex_local_minus2), KDim: (nlevp1, nlevp1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_39_40(
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    cell_endindex_local_minus1: int,
    nflatlev_startindex_plus1: int,
    nlev: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_39(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (2, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex_plus1, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfacq_c,
        out=w_concorr_c,
        domain={CellDim: (2, cell_endindex_local_minus1), KDim: (nlevp1, nlevp1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def corrector_stencils_35_39_40(
    vn: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    edge_endindex_local_minus2: int,
    cell_endindex_local_minus1: int,
    nflatlev_startindex_plus1: int,
    nlev: int,
    nlevp1: int,
    nflatlev_startindex: int,
):
    _mo_solve_nonhydro_stencil_35(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain={
            EdgeDim: (4, edge_endindex_local_minus2),
            KDim: (nflatlev_startindex, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_39(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (2, cell_endindex_local_minus1),
            KDim: (nflatlev_startindex_plus1, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfacq_c,
        out=w_concorr_c,
        domain={CellDim: (2, cell_endindex_local_minus1), KDim: (nlevp1, nlevp1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_42_44_45_45b(
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
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_interior: int,
    nlev: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_42(
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
        out=(z_w_expl, z_contr_w_fl_l),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (1, nlev),
        },
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_45(
        out=z_alpha,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (nlevp1, nlevp1),
        },
    )
    _mo_solve_nonhydro_stencil_45_b(
        out=z_q,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, 0),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_43_44_45_45b(
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
    cell_startindex_nudging_plus1: int,
    cell_endindex_interior: int,
    nlev: int,
    nlevp1: int,
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (1, nlev),
        },
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_45(
        out=z_alpha,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (nlevp1, nlevp1),
        },
    )
    _mo_solve_nonhydro_stencil_45_b(
        out=z_q,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, 0),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_47_48_49(
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
    cell_startindex_nudging_plus1: int,
    cell_endindex_interior: int,
    nlev: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_47(
        w_concorr_c,
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (nlevp1, nlevp1),
        },
    )
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, 0),
        },
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_52_53(
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
    cell_startindex_nudging_plus1: int,
    cell_endindex_interior: int,
    nlev: int,
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
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (1, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_53_scan(
        z_q,
        w,
        out=w[:, 1:],
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (1, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_59_60(
    exner_nnow: Field[[CellDim, KDim], float],
    exner_nnew: Field[[CellDim, KDim], float],
    exner_dyn_incr: Field[[CellDim, KDim], float],
    ddt_exner_phy: Field[[CellDim, KDim], float],
    ndyn_substeps_var: float,
    dtime: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_interior: int,
    kstart_moist: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_59(
        exner_nnow,
        out=exner_dyn_incr,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (kstart_moist, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_60(
        exner_nnew,
        ddt_exner_phy,
        exner_dyn_incr,
        ndyn_substeps_var,
        dtime,
        out=exner_dyn_incr,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_interior),
            KDim: (kstart_moist, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_61_62(
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
    cell_endindex_nudging: int,
    nlev: int,
    nlevp1: int,
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
        domain={CellDim: (0, cell_endindex_nudging), KDim: (0, nlev)},
    )
    _mo_solve_nonhydro_stencil_62(
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
        domain={CellDim: (0, cell_endindex_nudging), KDim: (nlevp1, nlevp1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_56_63(
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    cell_endindex_nudging: int,
    kstart_dd3d: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_56_63(
        inv_ddqz_z_full,
        w,
        w_concorr_c,
        out=z_dwdz_dd,
        domain={
            CellDim: (0, cell_endindex_nudging),
            KDim: (kstart_dd3d, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_66_67(
    bdy_halo_c: Field[[CellDim], bool],
    rho: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
    rd_o_p0ref: float,
    cell_startindex_interior_minus1: int,
    cell_endindex_local: int,
    cell_endindex_nudging: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_66(
        bdy_halo_c,
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        out=(theta_v, exner),
        domain={
            CellDim: (
                cell_startindex_interior_minus1,
                cell_endindex_local,
            ),  # TODO: @abishekg7 double check end index
            KDim: (0, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_67(
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        out=(theta_v, exner),
        domain={CellDim: (0, cell_endindex_nudging), KDim: (0, nlev)},
    )
