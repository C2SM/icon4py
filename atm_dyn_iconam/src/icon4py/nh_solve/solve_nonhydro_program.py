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

from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.atm_dyn_iconam.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_4th_order_divdamp import (
    mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_01 import (
    mo_solve_nonhydro_stencil_01,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_02 import (
    _mo_solve_nonhydro_stencil_02,
    mo_solve_nonhydro_stencil_02,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_03 import (
    mo_solve_nonhydro_stencil_03,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_12 import (
    _mo_solve_nonhydro_stencil_12,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_13 import (
    _mo_solve_nonhydro_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_14 import (
    _mo_solve_nonhydro_stencil_14,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_15 import (
    _mo_solve_nonhydro_stencil_15,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
    mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_18 import (
    _mo_solve_nonhydro_stencil_18,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_19 import (
    _mo_solve_nonhydro_stencil_19,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_22 import (
    _mo_solve_nonhydro_stencil_22,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_24 import (
    _mo_solve_nonhydro_stencil_24,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_28 import (
    _mo_solve_nonhydro_stencil_28,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_29 import (
    _mo_solve_nonhydro_stencil_29,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_32 import (
    _mo_solve_nonhydro_stencil_32,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_41 import (
    _mo_solve_nonhydro_stencil_41,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_46 import (
    _mo_solve_nonhydro_stencil_46,
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
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_50 import (
    _mo_solve_nonhydro_stencil_50,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_52 import (
    _mo_solve_nonhydro_stencil_52,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_53 import (
    _mo_solve_nonhydro_stencil_53_scan,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_54 import (
    _mo_solve_nonhydro_stencil_54,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_55 import (
    _mo_solve_nonhydro_stencil_55,
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
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_02(
        exner_exfac,
        exner,
        exner_ref_mc,
        exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={CellDim: (1, cell_endindex_local), KDim: (1, nlev)},
    )
    _set_zero_c_k(
        z_exner_ex_pr, domain={CellDim: (0, cell_endindex_local), KDim: (1, nlev)}
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_4_5_6(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    cell_endindex_local: int,
    nflatlev_startindex: int,
    nlev: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_04(
        wgtfacq_c,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={CellDim: (2, cell_endindex_local), KDim: (nlevp1, nlevp1)},
    )

    _mo_solve_nonhydro_stencil_05(
        wgtfac_c,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={
            CellDim: (2, cell_endindex_local),
            KDim: (maximum(1, nflatlev_startindex), nlev),
        },
    )

    _mo_solve_nonhydro_stencil_06(
        z_exner_ic,
        inv_ddqz_z_full,
        out=z_dexner_dz_c_1,
        domain={
            CellDim: (2, cell_endindex_local),
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
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_07(
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(z_rth_pr_1, z_rth_pr_2),
        domain={CellDim: (2, cell_endindex_local), KDim: (0, 0)},
    )
    _mo_solve_nonhydro_stencil_08(
        wgtfac_c,
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(rho_ic, z_rth_pr_1, z_rth_pr_2),
        domain={CellDim: (2, cell_endindex_local), KDim: (1, nlev)},
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
        domain={CellDim: (2, cell_endindex_local), KDim: (1, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def predictor_stencils_11_lower_upper(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    cell_endindex_local: int,
    nlevp1: int,
):
    _mo_solve_nonhydro_stencil_11_lower(
        out=z_theta_v_pr_ic, domain={CellDim: (2, cell_endindex_local), KDim: (0, 0)}
    )
    _mo_solve_nonhydro_stencil_11_upper(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={CellDim: (2, cell_endindex_local), KDim: (nlevp1, nlevp1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_12(
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_12(
        z_theta_v_pr_ic,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        z_rth_pr_2,
        out=z_dexner_dz_c_2,
        domain={},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_13(
    rho: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    cell_startindex_local_minus2: int,
    cell_endindex_local_minus2: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_13(
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(z_rth_pr_1, z_rth_pr_2),
        domain={
            CellDim: (cell_startindex_local_minus2, cell_endindex_local_minus2),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_14(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    edge_startindex_local_minus2: int,
    edge_endindex_local_minus3: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_14(
        out=(z_rho_e, z_theta_v_e),
        domain={
            EdgeDim: (edge_startindex_local_minus2, edge_endindex_local_minus3),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_15(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    edge_endindex_local_minus1: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_15(
        out=(z_rho_e, z_theta_v_e),
        domain={EdgeDim: (0, edge_endindex_local_minus1), KDim: (0, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    p_dthalf: float,
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_grad_rth_1: Field[[CellDim, KDim], float],
    z_grad_rth_2: Field[[CellDim, KDim], float],
    z_grad_rth_3: Field[[CellDim, KDim], float],
    z_grad_rth_4: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
        p_vn,
        p_vt,
        pos_on_tplane_e_1,
        pos_on_tplane_e_2,
        primal_normal_cell_1,
        dual_normal_cell_1,
        primal_normal_cell_2,
        dual_normal_cell_2,
        p_dthalf,
        rho_ref_me,
        theta_ref_me,
        z_grad_rth_1,
        z_grad_rth_2,
        z_grad_rth_3,
        z_grad_rth_4,
        z_rth_pr_1,
        z_rth_pr_2,
        out=(z_rho_e, z_theta_v_e),
        domain={},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_18(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    edge_startindex_nudging_plus1: int,
    edge_endindex_local: int,
    nflatlev_endindex_minus1: int,
):
    _mo_solve_nonhydro_stencil_18(
        inv_dual_edge_length,
        z_exner_ex_pr,
        out=z_gradh_exner,
        domain={
            EdgeDim: (edge_startindex_nudging_plus1, edge_endindex_local),
            KDim: (0, nflatlev_endindex_minus1),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def nhsolve_predictor_tendencies_19_20(
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_19(
        inv_dual_edge_length,
        z_exner_ex_pr,
        ddxn_z_full,
        c_lin_e,
        z_dexner_dz_c_1,
        out=z_gradh_exner,
        domain={},
    )


# TODO: @nfarabullini: what's up with stencil_20?


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_22(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    edge_startindex_nudging_plus1: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_22(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        out=z_gradh_exner,
        domain={
            EdgeDim: (edge_startindex_nudging_plus1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_24(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    vn_nnew: Field[[EdgeDim, KDim], float],
    dtime: float,
    cpd: float,
    edge_startindex_nudging_plus1: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_24(
        vn_nnow,
        ddt_vn_adv_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        cpd,
        out=vn_nnew,
        domain={
            EdgeDim: (edge_startindex_nudging_plus1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_28(
    vn_incr: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    iau_wgt_dyn: float,
):
    _mo_solve_nonhydro_stencil_28(vn_incr, vn, iau_wgt_dyn, out=vn, domain={})


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_29(
    grf_tend_vn: Field[[EdgeDim, KDim], float],
    vn_now: Field[[EdgeDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    dtime: float,
):
    _mo_solve_nonhydro_stencil_29(grf_tend_vn, vn_now, dtime, out=vn_new, domain={})


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_32(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_vn_avg: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    edge_endindex_local_minus2: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_32(
        z_rho_e,
        z_vn_avg,
        ddqz_z_full_e,
        z_theta_v_e,
        out=(mass_fl_e, z_theta_v_fl_e),
        domain={EdgeDim: (4, edge_endindex_local_minus2), KDim: (0, nlev)},
    )


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
    edge_endindex_local_minus2: int,
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
            EdgeDim: (4, edge_endindex_local_minus2),
            KDim: (nflatlev_startindex, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_36(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={EdgeDim: (4, edge_endindex_local_minus2), KDim: (1, nlev)},
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
def mo_solve_nonhydro_stencil_41(
    geofac_div: Field[[CellDim, C2EDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    z_flxdiv_mass: Field[[CellDim, KDim], float],
    z_flxdiv_theta: Field[[CellDim, KDim], float],
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_41(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        out=(z_flxdiv_mass, z_flxdiv_theta),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
        },
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
    cell_endindex_local: int,
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
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
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
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_45(
        out=z_alpha,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (nlevp1, nlevp1),
        },
    )
    _mo_solve_nonhydro_stencil_45_b(
        out=z_q,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
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
    cell_endindex_local: int,
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
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
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
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_45(
        out=z_alpha,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (nlevp1, nlevp1),
        },
    )
    _mo_solve_nonhydro_stencil_45_b(
        out=z_q,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, 0),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_46(
    w_nnew: Field[[CellDim, KDim], float],
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
):
    _mo_solve_nonhydro_stencil_46(
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
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
def mo_solve_nonhydro_stencil_50(
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    rho_incr: Field[[CellDim, KDim], float],
    exner_incr: Field[[CellDim, KDim], float],
    iau_wgt_dyn: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_50(
        z_rho_expl,
        z_exner_expl,
        rho_incr,
        exner_incr,
        iau_wgt_dyn,
        out=(z_rho_expl, z_exner_expl),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
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
    cell_startindex_nudging: int,
    cell_endindex_local: int,
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
            CellDim: (cell_startindex_nudging + 1, cell_endindex_local),
            KDim: (2, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_53_scan(
        z_q,
        w,
        out=w[:, 1:],
        domain={
            CellDim: (cell_startindex_nudging + 1, cell_endindex_local),
            KDim: (1, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_54(
    z_raylfac: Field[[KDim], float],
    w_1: Field[[CellDim], float],
    w: Field[[CellDim, KDim], float],
    cell_startindex_nudging_plus1,
    cell_endindex_local,
):
    _mo_solve_nonhydro_stencil_54(
        z_raylfac,
        w_1,
        w,
        out=w,
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_55(
    z_rho_expl: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    z_alpha: Field[[CellDim, KDim], float],
    z_beta: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    dtime: float,
    cvd_o_rd: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_55(
        z_rho_expl,
        vwind_impl_wgt,
        inv_ddqz_z_full,
        rho_ic,
        w,
        z_exner_expl,
        exner_ref_mc,
        z_alpha,
        z_beta,
        rho_now,
        theta_v_now,
        exner_now,
        dtime,
        cvd_o_rd,
        out=(rho_new, exner_new, theta_v_new),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
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
):
    _mo_solve_nonhydro_stencil_59(exner_nnow, out=exner_dyn_incr, domain={})
    _mo_solve_nonhydro_stencil_60(
        exner_nnew,
        ddt_exner_phy,
        exner_dyn_incr,
        ndyn_substeps_var,
        dtime,
        out=exner_dyn_incr,
        domain={},
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
):
    _mo_solve_nonhydro_stencil_56_63(
        inv_ddqz_z_full, w, w_concorr_c, out=z_dwdz_dd, domain={}
    )


@program(backend=gtfn_cpu.run_gtfn)
def stencils_66_67(
    bdy_halo_c: Field[[CellDim], bool],
    rho: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
    rd_o_p0ref: float,
    cell_endindex_nudging: int,
    cell_startindex_local: int,
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
            CellDim: (cell_startindex_local - 1, cell_startindex_local),
            KDim: (1, nlev),
        },
    )
    _mo_solve_nonhydro_stencil_67(
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        out=(theta_v, exner),
        domain={CellDim: (1, cell_endindex_nudging), KDim: (1, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_10(
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    rho_now: Field[[CellDim, KDim], float],
    rho_var: Field[[CellDim, KDim], float],
    theta_now: Field[[CellDim, KDim], float],
    theta_var: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    dtime: float,
    wgt_nnow_rth: float,
    wgt_nnew_rth: float,
    cell_startindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_10(
        w,
        w_concorr_c,
        ddqz_z_half,
        rho_now,
        rho_var,
        theta_now,
        theta_var,
        wgtfac_c,
        theta_ref_mc,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        out=(rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
        domain={CellDim: (2, cell_startindex_local), KDim: (1, nlev - 1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_17(
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    edge_endindex_local: int,
    kstart_dd3d: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_17(
        hmask_dd3d,
        scalfac_dd3d,
        inv_dual_edge_length,
        z_dwdz_dd,
        z_graddiv_vn,
        out=z_graddiv_vn,
        domain={CellDim: (6, edge_endindex_local - 2), KDim: (kstart_dd3d, nlev - 1)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_23(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl2: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    vn_nnew: Field[[EdgeDim, KDim], float],
    dtime: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    cpd: float,
    edge_startindex_nudging: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_23(
        vn_nnow,
        ddt_vn_adv_ntl1,
        ddt_vn_adv_ntl2,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        out=vn_nnew,
        domain={
            CellDim: (edge_startindex_nudging + 1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_25(
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    edge_startindex_nudging: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_25(
        geofac_grdiv,
        z_graddiv_vn,
        out=z_graddiv2_vn,
        domain={
            CellDim: (edge_startindex_nudging + 1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_26(
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    scal_divdamp_o2: float,
    edge_startindex_nudging: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_26(
        z_graddiv_vn,
        vn,
        scal_divdamp_o2,
        out=vn,
        domain={
            CellDim: (edge_startindex_nudging + 1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_27(
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    edge_startindex_nudging: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_27(
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            CellDim: (edge_startindex_nudging + 1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_4th_order_divdamp(
    scal_divdamp: Field[[KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    edge_startindex_nudging: int,
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_4th_order_divdamp(
        scal_divdamp,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            CellDim: (edge_startindex_nudging + 1, edge_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_33(
    vn_traj: Field[[EdgeDim, KDim], float],
    mass_flx_me: Field[[EdgeDim, KDim], float],
    edge_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_33(
        out=(vn_traj, mass_flx_me),
        domain={EdgeDim: (0, edge_endindex_local), KDim: (0, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_34(
    z_vn_avg: Field[[EdgeDim, KDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    vn_traj: Field[[EdgeDim, KDim], float],
    mass_flx_me: Field[[EdgeDim, KDim], float],
    r_nsubsteps: float,
    edge_endindex_local_minus2: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_34(
        z_vn_avg,
        mass_fl_e,
        vn_traj,
        mass_flx_me,
        r_nsubsteps,
        out=(vn_traj, mass_flx_me),
        domain={EdgeDim: (4, edge_endindex_local_minus2), KDim: (0, nlev)},
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_46(
    w_nnew: Field[[CellDim, KDim], float],
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
):
    _mo_solve_nonhydro_stencil_46(
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, 0),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_58(
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_58(
        z_contr_w_fl_l,
        rho_ic,
        vwind_impl_wgt,
        w,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
        domain={
            EdgeDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
        },
    )


@program(backend=gtfn_cpu.run_gtfn)
def mo_solve_nonhydro_stencil_65(
    rho_ic: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w_now: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
    cell_startindex_nudging_plus1: int,
    cell_endindex_local: int,
    nlev: int,
):
    _mo_solve_nonhydro_stencil_65(
        rho_ic,
        vwind_expl_wgt,
        vwind_impl_wgt,
        w_now,
        w_new,
        w_concorr_c,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
        domain={
            EdgeDim: (cell_startindex_nudging_plus1, cell_endindex_local),
            KDim: (0, nlev),
        },
    )
