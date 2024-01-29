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
from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, where

from icon4py.model.atmosphere.dycore.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.compute_first_vertical_derivative import (
    _compute_first_vertical_derivative,
)
from icon4py.model.atmosphere.dycore.compute_pertubation_of_rho_and_theta import (
    _compute_pertubation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_pertubation_of_rho_and_theta_and_rho_at_ic import (
    _compute_pertubation_of_rho_and_theta_and_rho_at_ic,
)
from icon4py.model.atmosphere.dycore.compute_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta import (
    _compute_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta,
)
from icon4py.model.atmosphere.dycore.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure,
)
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.atmosphere.dycore.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_11_upper import (
    _mo_solve_nonhydro_stencil_11_upper,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_36 import (
    _mo_solve_nonhydro_stencil_36,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_37 import (
    _mo_solve_nonhydro_stencil_37,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_39 import (
    _mo_solve_nonhydro_stencil_39,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_40 import (
    _mo_solve_nonhydro_stencil_40,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_42 import (
    _mo_solve_nonhydro_stencil_42,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_43 import (
    _mo_solve_nonhydro_stencil_43,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_44 import (
    _mo_solve_nonhydro_stencil_44,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_47 import (
    _mo_solve_nonhydro_stencil_47,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_49 import (
    _mo_solve_nonhydro_stencil_49,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_61 import (
    _mo_solve_nonhydro_stencil_61,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_62 import (
    _mo_solve_nonhydro_stencil_62,
)
from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_vp import (
    _set_cell_kdim_field_to_zero_vp,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _set_zero_c_k, _set_zero_e_k
from icon4py.model.common.dimension import CEDim, CellDim, ECDim, EdgeDim, KDim


# TODO: abishekg7 move this to tests
@program
def init_test_fields(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    indices_edges_1: int32,
    indices_edges_2: int32,
    indices_cells_1: int32,
    indices_cells_2: int32,
    nlev: int32,
):
    _set_zero_e_k(
        out=z_rho_e,
        domain={EdgeDim: (indices_edges_1, indices_edges_2), KDim: (0, nlev)},
    )
    _set_zero_e_k(
        out=z_theta_v_e,
        domain={EdgeDim: (indices_edges_1, indices_edges_2), KDim: (0, nlev)},
    )
    _set_zero_e_k(
        out=z_graddiv_vn,
        domain={EdgeDim: (indices_edges_1, indices_edges_2), KDim: (0, nlev)},
    )
    _set_zero_c_k(
        out=z_dwdz_dd,
        domain={CellDim: (indices_cells_1, indices_cells_2), KDim: (0, nlev)},
    )


@field_operator
def _predictor_stencils_2_3(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    (z_exner_ex_pr, exner_pr) = where(
        (k_field >= int32(0)) & (k_field < nlev),
        _extrapolate_temporally_exner_pressure(exner_exfac, exner, exner_ref_mc, exner_pr),
        (z_exner_ex_pr, exner_pr),
    )
    z_exner_ex_pr = where(k_field == nlev, _set_zero_c_k(), z_exner_ex_pr)

    return z_exner_ex_pr, exner_pr


@program
def predictor_stencils_2_3(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _predictor_stencils_2_3(
        exner_exfac,
        exner,
        exner_ref_mc,
        exner_pr,
        z_exner_ex_pr,
        k_field,
        nlev,
        out=(z_exner_ex_pr, exner_pr),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _predictor_stencils_4_5_6(
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    # Perturbation Exner pressure on bottom half level
    z_exner_ic = where(
        k_field == nlev,
        _interpolate_to_surface(wgtfacq_c_dsl, z_exner_ex_pr),
        z_exner_ic,
    )

    # WS: moved full z_exner_ic calculation here to avoid OpenACC dependency on jk+1 below
    # possibly GZ will want to consider the cache ramifications of this change for CPU
    z_exner_ic = where(
        k_field < nlev,
        _interpolate_to_half_levels_vp(wgtfac_c, z_exner_ex_pr),
        z_exner_ic,
    )

    # First vertical derivative of perturbation Exner pressure
    z_dexner_dz_c_1 = where(
        k_field < nlev,
        _compute_first_vertical_derivative(z_exner_ic, inv_ddqz_z_full),
        z_dexner_dz_c_1,
    )
    return z_exner_ic, z_dexner_dz_c_1


@program
def predictor_stencils_4_5_6(
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_exner_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _predictor_stencils_4_5_6(
        wgtfacq_c_dsl,
        z_exner_ex_pr,
        z_exner_ic,
        wgtfac_c,
        inv_ddqz_z_full,
        z_dexner_dz_c_1,
        k_field,
        nlev,
        out=(z_exner_ic, z_dexner_dz_c_1),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _predictor_stencils_7_8_9(
    rho: Field[[CellDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    rho_ref_mc: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (z_rth_pr_1, z_rth_pr_2) = where(
        k_field == int32(0),
        _compute_pertubation_of_rho_and_theta(rho, rho_ref_mc, theta_v, theta_ref_mc),
        (z_rth_pr_1, z_rth_pr_2),
    )

    (rho_ic, z_rth_pr_1, z_rth_pr_2) = where(
        k_field >= int32(1),
        _compute_pertubation_of_rho_and_theta_and_rho_at_ic(
            wgtfac_c, rho, rho_ref_mc, theta_v, theta_ref_mc
        ),
        (rho_ic, z_rth_pr_1, z_rth_pr_2),
    )

    (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        k_field >= int32(1),
        _compute_theta_v_prime_at_ic_and_theta_v_at_ic_and_pressure_gradient_theta(
            wgtfac_c,
            z_rth_pr_2,
            theta_v,
            vwind_expl_wgt,
            exner_pr,
            d_exner_dz_ref_ic,
            ddqz_z_half,
        ),
        (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
    )

    return z_rth_pr_1, z_rth_pr_2, rho_ic, z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c


@program
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
    k_field: Field[[KDim], int32],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _predictor_stencils_7_8_9(
        rho,
        z_rth_pr_1,
        z_rth_pr_2,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        rho_ic,
        wgtfac_c,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
        k_field,
        nlev,
        out=(
            z_rth_pr_1,
            z_rth_pr_2,
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _predictor_stencils_11_lower_upper(
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_theta_v_pr_ic = where(k_field == int32(0), _set_cell_kdim_field_to_zero_vp(), z_theta_v_pr_ic)

    (z_theta_v_pr_ic, theta_v_ic) = where(
        k_field == nlev,
        _mo_solve_nonhydro_stencil_11_upper(wgtfacq_c_dsl, z_rth_pr, theta_ref_ic),
        (z_theta_v_pr_ic, theta_v_ic),
    )
    return z_theta_v_pr_ic, theta_v_ic


@program
def predictor_stencils_11_lower_upper(
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _predictor_stencils_11_lower_upper(
        wgtfacq_c_dsl,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        theta_v_ic,
        k_field,
        nlev,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED)
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
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _predictor_stencils_35_36(
    vn: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    z_w_concorr_me = where(
        k_field >= nflatlev_startindex,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )
    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k_field >= int32(1),
        _mo_solve_nonhydro_stencil_36(wgtfac_e, vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )
    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@program
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
    k_field: Field[[KDim], int32],
    nflatlev_startindex: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _predictor_stencils_35_36(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        z_w_concorr_me,
        wgtfac_e,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
        k_field,
        nflatlev_startindex,
        out=(z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program
def predictor_stencils_37_38(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    wgtfacq_e_dsl: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_37(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_start + 1),
        },
    )
    _extrapolate_at_top(
        vn,
        wgtfacq_e_dsl,
        out=vn_ie,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )


@field_operator
def _stencils_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex_plus1: int32,
    nlev: int32,
) -> Field[[CellDim, KDim], float]:
    w_concorr_c = where(
        k_field >= nflatlev_startindex_plus1,  # TODO: @abishekg7 does this need to change
        _mo_solve_nonhydro_stencil_39(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        w_concorr_c,
    )

    w_concorr_c = where(
        k_field == nlev,
        _mo_solve_nonhydro_stencil_40(e_bln_c_s, z_w_concorr_me, wgtfacq_c_dsl),
        w_concorr_c,
    )

    return w_concorr_c


@program
def stencils_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c_dsl: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    nflatlev_startindex_plus1: int32,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _stencils_39_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        wgtfacq_c_dsl,
        w_concorr_c,
        k_field,
        nflatlev_startindex_plus1,
        nlev,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _stencils_42_44_45_45b(
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
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (z_w_expl, z_contr_w_fl_l) = where(
        (k_field >= int32(1)) & (k_field < nlev),
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
        ),
        (z_w_expl, z_contr_w_fl_l),
    )

    (z_beta, z_alpha) = where(
        (k_field >= int32(0)) & (k_field < nlev),
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
        ),
        (z_beta, z_alpha),
    )
    z_alpha = where(k_field == nlev, _set_cell_kdim_field_to_zero_vp(), z_alpha)

    z_q = where(k_field == int32(0), _set_cell_kdim_field_to_zero_vp(), z_q)
    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@program
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
    k_field: Field[[KDim], int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _stencils_42_44_45_45b(
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
        rd,
        cvd,
        dtime,
        cpd,
        wgt_nnow_vel,
        wgt_nnew_vel,
        nlev,
        out=(z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _stencils_43_44_45_45b(
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
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    nlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (z_w_expl, z_contr_w_fl_l) = where(
        (k_field >= int32(1)) & (k_field < nlev),
        _mo_solve_nonhydro_stencil_43(
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
        (k_field >= int32(0)) & (k_field < nlev),
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
        ),
        (z_beta, z_alpha),
    )
    z_alpha = where(k_field == nlev, _set_cell_kdim_field_to_zero_vp(), z_alpha)
    z_q = where(k_field == int32(0), _set_cell_kdim_field_to_zero_vp(), z_q)

    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@program
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
    k_field: Field[[KDim], int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _stencils_43_44_45_45b(
        z_w_expl,
        w_nnow,
        ddt_w_adv_ntl1,
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
        rd,
        cvd,
        dtime,
        cpd,
        nlev,
        out=(z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _stencils_47_48_49(
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
    k_field: Field[[KDim], int32],
    dtime: float,
    nlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (w_nnew, z_contr_w_fl_l) = where(
        k_field == nlev,
        _mo_solve_nonhydro_stencil_47(w_concorr_c),
        (w_nnew, z_contr_w_fl_l),
    )
    # 48 and 49 are identical except for bounds
    (z_rho_expl, z_exner_expl) = where(
        (k_field >= int32(0)) & (k_field < nlev),
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
        ),
        (z_rho_expl, z_exner_expl),
    )
    return w_nnew, z_contr_w_fl_l, z_rho_expl, z_exner_expl


@program(grid_type=GridType.UNSTRUCTURED)
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
    k_field: Field[[KDim], int32],
    dtime: float,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_end: int32,
    vertical_start: int32,
):
    _stencils_47_48_49(
        w_nnew,
        z_contr_w_fl_l,
        w_concorr_c,
        z_rho_expl,
        z_exner_expl,
        rho_nnow,
        inv_ddqz_z_full,
        z_flxdiv_mass,
        exner_pr,
        z_beta,
        z_flxdiv_theta,
        theta_v_ic,
        ddt_exner_phy,
        k_field,
        dtime,
        nlev,
        out=(w_nnew, z_contr_w_fl_l, z_rho_expl, z_exner_expl),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _stencils_61_62(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    dtime: float,
    nlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    (rho_new, exner_new, w_new) = where(
        (k_field >= int32(0)) & (k_field < nlev),
        _mo_solve_nonhydro_stencil_61(
            rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w, dtime
        ),
        (rho_new, exner_new, w_new),
    )
    w_new = where(
        k_field == nlev,
        _mo_solve_nonhydro_stencil_62(w_now, grf_tend_w, dtime),
        w_new,
    )
    return rho_new, exner_new, w_new


@program(grid_type=GridType.UNSTRUCTURED)
def stencils_61_62(
    rho_now: Field[[CellDim, KDim], float],
    grf_tend_rho: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    grf_tend_thv: Field[[CellDim, KDim], float],
    w_now: Field[[CellDim, KDim], float],
    grf_tend_w: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    k_field: Field[[KDim], int32],
    dtime: float,
    nlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _stencils_61_62(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        rho_new,
        exner_new,
        w_new,
        k_field,
        dtime,
        nlev,
        out=(rho_new, exner_new, w_new),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
