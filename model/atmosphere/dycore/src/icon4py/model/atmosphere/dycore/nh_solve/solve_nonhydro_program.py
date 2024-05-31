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
from model.common.tests import field_type_aliases as fa

from icon4py.model.atmosphere.dycore.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
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
from icon4py.model.atmosphere.dycore.compute_first_vertical_derivative import (
    _compute_first_vertical_derivative,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers,
)
from icon4py.model.atmosphere.dycore.compute_solver_coefficients_matrix import (
    _compute_solver_coefficients_matrix,
)
from icon4py.model.atmosphere.dycore.compute_virtual_potential_temperatures_and_pressure_gradient import (
    _compute_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure,
)
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.atmosphere.dycore.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.atmosphere.dycore.interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges import (
    _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.set_lower_boundary_condition_for_w_and_contravariant_correction import (
    _set_lower_boundary_condition_for_w_and_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.set_theta_v_prime_ic_at_lower_boundary import (
    _set_theta_v_prime_ic_at_lower_boundary,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import (
    _broadcast_zero_to_three_edge_kdim_fields_wp,
)
from icon4py.model.atmosphere.dycore.update_density_exner_wind import _update_density_exner_wind
from icon4py.model.atmosphere.dycore.update_wind import _update_wind
from icon4py.model.common.dimension import CEDim, CellDim, ECDim, EdgeDim, KDim
from icon4py.model.common.settings import backend


# TODO: abishekg7 move this to tests
@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_test_fields(
    z_rho_e: fa.EKfloatField,
    z_theta_v_e: fa.EKfloatField,
    z_dwdz_dd: fa.CKfloatField,
    z_graddiv_vn: fa.EKfloatField,
    indices_edges_1: int32,
    indices_edges_2: int32,
    indices_cells_1: int32,
    indices_cells_2: int32,
    nlev: int32,
):
    _broadcast_zero_to_three_edge_kdim_fields_wp(
        out=(z_rho_e, z_theta_v_e, z_graddiv_vn),
        domain={EdgeDim: (indices_edges_1, indices_edges_2), KDim: (0, nlev)},
    )
    _init_cell_kdim_field_with_zero_wp(
        out=z_dwdz_dd,
        domain={CellDim: (indices_cells_1, indices_cells_2), KDim: (0, nlev)},
    )


@field_operator
def _predictor_stencils_2_3(
    exner_exfac: fa.CKfloatField,
    exner: fa.CKfloatField,
    exner_ref_mc: fa.CKfloatField,
    exner_pr: fa.CKfloatField,
    z_exner_ex_pr: fa.CKfloatField,
    k_field: fa.KintField,
    nlev: int32,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    (z_exner_ex_pr, exner_pr) = where(
        (k_field >= 0) & (k_field < nlev),
        _extrapolate_temporally_exner_pressure(exner_exfac, exner, exner_ref_mc, exner_pr),
        (z_exner_ex_pr, exner_pr),
    )
    z_exner_ex_pr = where(k_field == nlev, _init_cell_kdim_field_with_zero_wp(), z_exner_ex_pr)

    return z_exner_ex_pr, exner_pr


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_2_3(
    exner_exfac: fa.CKfloatField,
    exner: fa.CKfloatField,
    exner_ref_mc: fa.CKfloatField,
    exner_pr: fa.CKfloatField,
    z_exner_ex_pr: fa.CKfloatField,
    k_field: fa.KintField,
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
    wgtfacq_c_dsl: fa.CKfloatField,
    z_exner_ex_pr: fa.CKfloatField,
    z_exner_ic: fa.CKfloatField,
    wgtfac_c: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_dexner_dz_c_1: fa.CKfloatField,
    k_field: fa.KintField,
    nlev: int32,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
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
        _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_exner_ex_pr),
        z_exner_ic,
    )

    # First vertical derivative of perturbation Exner pressure
    z_dexner_dz_c_1 = where(
        k_field < nlev,
        _compute_first_vertical_derivative(z_exner_ic, inv_ddqz_z_full),
        z_dexner_dz_c_1,
    )
    return z_exner_ic, z_dexner_dz_c_1


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_4_5_6(
    wgtfacq_c_dsl: fa.CKfloatField,
    z_exner_ex_pr: fa.CKfloatField,
    z_exner_ic: fa.CKfloatField,
    wgtfac_c: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_dexner_dz_c_1: fa.CKfloatField,
    k_field: fa.KintField,
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
    rho: fa.CKfloatField,
    z_rth_pr_1: fa.CKfloatField,
    z_rth_pr_2: fa.CKfloatField,
    rho_ref_mc: fa.CKfloatField,
    theta_v: fa.CKfloatField,
    theta_ref_mc: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    wgtfac_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    exner_pr: fa.CKfloatField,
    d_exner_dz_ref_ic: fa.CKfloatField,
    ddqz_z_half: fa.CKfloatField,
    z_theta_v_pr_ic: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    k_field: fa.KintField,
    nlev: int32,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
]:
    (z_rth_pr_1, z_rth_pr_2) = where(
        k_field == 0,
        _compute_perturbation_of_rho_and_theta(rho, rho_ref_mc, theta_v, theta_ref_mc),
        (z_rth_pr_1, z_rth_pr_2),
    )

    (rho_ic, z_rth_pr_1, z_rth_pr_2) = where(
        k_field >= 1,
        _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
            wgtfac_c, rho, rho_ref_mc, theta_v, theta_ref_mc
        ),
        (rho_ic, z_rth_pr_1, z_rth_pr_2),
    )

    (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = where(
        k_field >= 1,
        _compute_virtual_potential_temperatures_and_pressure_gradient(
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_7_8_9(
    rho: fa.CKfloatField,
    rho_ref_mc: fa.CKfloatField,
    theta_v: fa.CKfloatField,
    theta_ref_mc: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    z_rth_pr_1: fa.CKfloatField,
    z_rth_pr_2: fa.CKfloatField,
    wgtfac_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    exner_pr: fa.CKfloatField,
    d_exner_dz_ref_ic: fa.CKfloatField,
    ddqz_z_half: fa.CKfloatField,
    z_theta_v_pr_ic: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    k_field: fa.KintField,
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
    wgtfacq_c_dsl: fa.CKfloatField,
    z_rth_pr: fa.CKfloatField,
    theta_ref_ic: fa.CKfloatField,
    z_theta_v_pr_ic: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    k_field: fa.KintField,
    nlev: int32,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    z_theta_v_pr_ic = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_theta_v_pr_ic)

    (z_theta_v_pr_ic, theta_v_ic) = where(
        k_field == nlev,
        _set_theta_v_prime_ic_at_lower_boundary(wgtfacq_c_dsl, z_rth_pr, theta_ref_ic),
        (z_theta_v_pr_ic, theta_v_ic),
    )
    return z_theta_v_pr_ic, theta_v_ic


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_11_lower_upper(
    wgtfacq_c_dsl: fa.CKfloatField,
    z_rth_pr: fa.CKfloatField,
    theta_ref_ic: fa.CKfloatField,
    z_theta_v_pr_ic: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    k_field: fa.KintField,
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_advection_of_rho_and_theta(
    p_vn: fa.EKfloatField,
    p_vt: fa.EKfloatField,
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    p_dthalf: float,
    rho_ref_me: fa.EKfloatField,
    theta_ref_me: fa.EKfloatField,
    z_grad_rth_1: fa.CKfloatField,
    z_grad_rth_2: fa.CKfloatField,
    z_grad_rth_3: fa.CKfloatField,
    z_grad_rth_4: fa.CKfloatField,
    z_rth_pr_1: fa.CKfloatField,
    z_rth_pr_2: fa.CKfloatField,
    z_rho_e: fa.EKfloatField,
    z_theta_v_e: fa.EKfloatField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_advection_of_rho_and_theta(
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
    vn: fa.EKfloatField,
    ddxn_z_full: fa.EKfloatField,
    ddxt_z_full: fa.EKfloatField,
    vt: fa.EKfloatField,
    z_w_concorr_me: fa.EKfloatField,
    wgtfac_e: fa.EKfloatField,
    vn_ie: fa.EKfloatField,
    z_vt_ie: fa.EKfloatField,
    z_kin_hor_e: fa.EKfloatField,
    k_field: fa.KintField,
    nflatlev_startindex: int32,
) -> tuple[
    fa.EKfloatField,
    fa.EKfloatField,
    fa.EKfloatField,
    fa.EKfloatField,
]:
    z_w_concorr_me = where(
        k_field >= nflatlev_startindex,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )
    (vn_ie, z_vt_ie, z_kin_hor_e) = where(
        k_field >= 1,
        _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(wgtfac_e, vn, vt),
        (vn_ie, z_vt_ie, z_kin_hor_e),
    )
    return z_w_concorr_me, vn_ie, z_vt_ie, z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_35_36(
    vn: fa.EKfloatField,
    ddxn_z_full: fa.EKfloatField,
    ddxt_z_full: fa.EKfloatField,
    vt: fa.EKfloatField,
    z_w_concorr_me: fa.EKfloatField,
    wgtfac_e: fa.EKfloatField,
    vn_ie: fa.EKfloatField,
    z_vt_ie: fa.EKfloatField,
    z_kin_hor_e: fa.EKfloatField,
    k_field: fa.KintField,
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def predictor_stencils_37_38(
    vn: fa.EKfloatField,
    vt: fa.EKfloatField,
    vn_ie: fa.EKfloatField,
    z_vt_ie: fa.EKfloatField,
    z_kin_hor_e: fa.EKfloatField,
    wgtfacq_e_dsl: fa.EKfloatField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_kinetic_energy(
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
    z_w_concorr_me: fa.EKfloatField,
    wgtfac_c: fa.CKfloatField,
    wgtfacq_c_dsl: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    k_field: fa.KintField,
    nflatlev_startindex_plus1: int32,
    nlev: int32,
) -> fa.CKfloatField:
    w_concorr_c = where(
        k_field >= nflatlev_startindex_plus1,  # TODO: @abishekg7 does this need to change
        _compute_contravariant_correction_of_w(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        w_concorr_c,
    )

    w_concorr_c = where(
        k_field == nlev,
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, z_w_concorr_me, wgtfacq_c_dsl
        ),
        w_concorr_c,
    )

    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def stencils_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: fa.EKfloatField,
    wgtfac_c: fa.CKfloatField,
    wgtfacq_c_dsl: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    k_field: fa.KintField,
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
    z_w_expl: fa.CKfloatField,
    w_nnow: fa.CKfloatField,
    ddt_w_adv_ntl1: fa.CKfloatField,
    ddt_w_adv_ntl2: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    z_beta: fa.CKfloatField,
    exner_nnow: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    theta_v_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    z_q: fa.CKfloatField,
    k_field: fa.KintField,
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: int32,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
]:
    (z_w_expl, z_contr_w_fl_l) = where(
        (k_field >= 1) & (k_field < nlev),
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

    (z_beta, z_alpha) = where(
        (k_field >= 0) & (k_field < nlev),
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
    z_alpha = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), z_alpha)

    z_q = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_q)
    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def stencils_42_44_45_45b(
    z_w_expl: fa.CKfloatField,
    w_nnow: fa.CKfloatField,
    ddt_w_adv_ntl1: fa.CKfloatField,
    ddt_w_adv_ntl2: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    z_beta: fa.CKfloatField,
    exner_nnow: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    theta_v_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    z_q: fa.CKfloatField,
    k_field: fa.KintField,
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
    z_w_expl: fa.CKfloatField,
    w_nnow: fa.CKfloatField,
    ddt_w_adv_ntl1: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    z_beta: fa.CKfloatField,
    exner_nnow: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    theta_v_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    z_q: fa.CKfloatField,
    k_field: fa.KintField,
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    nlev: int32,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
]:
    (z_w_expl, z_contr_w_fl_l) = where(
        (k_field >= 1) & (k_field < nlev),
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
        (k_field >= 0) & (k_field < nlev),
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
    z_alpha = where(k_field == nlev, _init_cell_kdim_field_with_zero_vp(), z_alpha)
    z_q = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_q)

    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def stencils_43_44_45_45b(
    z_w_expl: fa.CKfloatField,
    w_nnow: fa.CKfloatField,
    ddt_w_adv_ntl1: fa.CKfloatField,
    z_th_ddz_exner_c: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    rho_ic: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    vwind_expl_wgt: fa.CfloatField,
    z_beta: fa.CKfloatField,
    exner_nnow: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    theta_v_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    z_q: fa.CKfloatField,
    k_field: fa.KintField,
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
    w_nnew: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    z_rho_expl: fa.CKfloatField,
    z_exner_expl: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_flxdiv_mass: fa.CKfloatField,
    exner_pr: fa.CKfloatField,
    z_beta: fa.CKfloatField,
    z_flxdiv_theta: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    ddt_exner_phy: fa.CKfloatField,
    k_field: fa.KintField,
    dtime: float,
    nlev: int32,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
]:
    (w_nnew, z_contr_w_fl_l) = where(
        k_field == nlev,
        _set_lower_boundary_condition_for_w_and_contravariant_correction(w_concorr_c),
        (w_nnew, z_contr_w_fl_l),
    )
    # 48 and 49 are identical except for bounds
    (z_rho_expl, z_exner_expl) = where(
        (k_field >= 0) & (k_field < nlev),
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
    return w_nnew, z_contr_w_fl_l, z_rho_expl, z_exner_expl


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def stencils_47_48_49(
    w_nnew: fa.CKfloatField,
    z_contr_w_fl_l: fa.CKfloatField,
    w_concorr_c: fa.CKfloatField,
    z_rho_expl: fa.CKfloatField,
    z_exner_expl: fa.CKfloatField,
    rho_nnow: fa.CKfloatField,
    inv_ddqz_z_full: fa.CKfloatField,
    z_flxdiv_mass: fa.CKfloatField,
    exner_pr: fa.CKfloatField,
    z_beta: fa.CKfloatField,
    z_flxdiv_theta: fa.CKfloatField,
    theta_v_ic: fa.CKfloatField,
    ddt_exner_phy: fa.CKfloatField,
    k_field: fa.KintField,
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
    rho_now: fa.CKfloatField,
    grf_tend_rho: fa.CKfloatField,
    theta_v_now: fa.CKfloatField,
    grf_tend_thv: fa.CKfloatField,
    w_now: fa.CKfloatField,
    grf_tend_w: fa.CKfloatField,
    rho_new: fa.CKfloatField,
    exner_new: fa.CKfloatField,
    w_new: fa.CKfloatField,
    k_field: fa.KintField,
    dtime: float,
    nlev: int32,
) -> tuple[
    fa.CKfloatField,
    fa.CKfloatField,
    fa.CKfloatField,
]:
    (rho_new, exner_new, w_new) = where(
        (k_field >= 0) & (k_field < nlev),
        _update_density_exner_wind(
            rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w, dtime
        ),
        (rho_new, exner_new, w_new),
    )
    w_new = where(
        k_field == nlev,
        _update_wind(w_now, grf_tend_w, dtime),
        w_new,
    )
    return rho_new, exner_new, w_new


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def stencils_61_62(
    rho_now: fa.CKfloatField,
    grf_tend_rho: fa.CKfloatField,
    theta_v_now: fa.CKfloatField,
    grf_tend_thv: fa.CKfloatField,
    w_now: fa.CKfloatField,
    grf_tend_w: fa.CKfloatField,
    rho_new: fa.CKfloatField,
    exner_new: fa.CKfloatField,
    w_new: fa.CKfloatField,
    k_field: fa.KintField,
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
