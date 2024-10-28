# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import where

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
from icon4py.model.atmosphere.dycore.update_density_exner_wind import (
    _update_density_exner_wind,
)
from icon4py.model.atmosphere.dycore.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa


# TODO: abishekg7 move this to tests
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_test_fields(
    z_rho_e: fa.EdgeKField[float],
    z_theta_v_e: fa.EdgeKField[float],
    z_dwdz_dd: fa.CellKField[float],
    z_graddiv_vn: fa.EdgeKField[float],
    edges_start: gtx.int32,
    edges_end: gtx.int32,
    cells_start: gtx.int32,
    cells_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _broadcast_zero_to_three_edge_kdim_fields_wp(
        out=(z_rho_e, z_theta_v_e, z_graddiv_vn),
        domain={dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
    )
    _init_cell_kdim_field_with_zero_wp(
        out=z_dwdz_dd,
        domain={dims.CellDim: (cells_start, cells_end), dims.KDim: (vertical_start, vertical_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_stencils_2_3(
    exner_exfac: fa.CellKField[float],
    exner: fa.CellKField[float],
    exner_ref_mc: fa.CellKField[float],
    exner_pr: fa.CellKField[float],
    z_exner_ex_pr: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    FIXME:
        - The first operation on z_exner_ex_pr should be done in a generic
          math (1+a)*x - a*y program
        - In the stencil, _extrapolate_temporally_exner_pressure doesn't only
          do what the name suggests: it also updates exner_pr, which is not
          what the name implies.
    """
    _extrapolate_temporally_exner_pressure(
        exner_exfac,
        exner,
        exner_ref_mc,
        exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _init_cell_kdim_field_with_zero_wp(
        out=z_exner_ex_pr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_stencils_4_5_6(
    wgtfacq_c_dsl: fa.CellKField[float],
    z_exner_ex_pr: fa.CellKField[float],
    z_exner_ic: fa.CellKField[float],
    wgtfac_c: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_dexner_dz_c_1: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    FIXME:
        - The value of z_exner_ic at the model top level is not updated
          and assumed to be zero. It should be treated in the same way as
          the ground level.
    """
    _interpolate_to_surface(
        wgtfacq_c_dsl,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
    _interpolate_to_half_levels_vp(
        wgtfac_c,
        z_exner_ex_pr,
        out=z_exner_ic,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _compute_first_vertical_derivative(
        z_exner_ic,
        inv_ddqz_z_full,
        out=z_dexner_dz_c_1,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )


@gtx.field_operator
def _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
    rho: fa.CellKField[float],
    z_rth_pr_1: fa.CellKField[float],
    z_rth_pr_2: fa.CellKField[float],
    rho_ref_mc: fa.CellKField[float],
    theta_v: fa.CellKField[float],
    theta_ref_mc: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    wgtfac_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    exner_pr: fa.CellKField[float],
    d_exner_dz_ref_ic: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    z_theta_v_pr_ic: fa.CellKField[float],
    theta_v_ic: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
    rho: fa.CellKField[float],
    rho_ref_mc: fa.CellKField[float],
    theta_v: fa.CellKField[float],
    theta_ref_mc: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    z_rth_pr_1: fa.CellKField[float],
    z_rth_pr_2: fa.CellKField[float],
    wgtfac_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    exner_pr: fa.CellKField[float],
    d_exner_dz_ref_ic: fa.CellKField[float],
    ddqz_z_half: fa.CellKField[float],
    z_theta_v_pr_ic: fa.CellKField[float],
    theta_v_ic: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
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
        out=(
            z_rth_pr_1,
            z_rth_pr_2,
            rho_ic,
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _predictor_stencils_11_lower_upper(
    wgtfacq_c_dsl: fa.CellKField[float],
    z_rth_pr: fa.CellKField[float],
    theta_ref_ic: fa.CellKField[float],
    z_theta_v_pr_ic: fa.CellKField[float],
    theta_v_ic: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nlev: gtx.int32,
) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
    z_theta_v_pr_ic = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_theta_v_pr_ic)

    (z_theta_v_pr_ic, theta_v_ic) = where(
        k_field == nlev,
        _set_theta_v_prime_ic_at_lower_boundary(wgtfacq_c_dsl, z_rth_pr, theta_ref_ic),
        (z_theta_v_pr_ic, theta_v_ic),
    )
    return z_theta_v_pr_ic, theta_v_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_stencils_11_lower_upper(
    wgtfacq_c_dsl: fa.CellKField[float],
    z_rth_pr: fa.CellKField[float],
    theta_ref_ic: fa.CellKField[float],
    z_theta_v_pr_ic: fa.CellKField[float],
    theta_v_ic: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_advection_of_rho_and_theta(
    p_vn: fa.EdgeKField[float],
    p_vt: fa.EdgeKField[float],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.ECDim], float],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.ECDim], float],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.ECDim], float],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.ECDim], float],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.ECDim], float],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.ECDim], float],
    p_dthalf: float,
    rho_ref_me: fa.EdgeKField[float],
    theta_ref_me: fa.EdgeKField[float],
    z_grad_rth_1: fa.CellKField[float],
    z_grad_rth_2: fa.CellKField[float],
    z_grad_rth_3: fa.CellKField[float],
    z_grad_rth_4: fa.CellKField[float],
    z_rth_pr_1: fa.CellKField[float],
    z_rth_pr_2: fa.CellKField[float],
    z_rho_e: fa.EdgeKField[float],
    z_theta_v_e: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _predictor_stencils_35_36(
    vn: fa.EdgeKField[float],
    ddxn_z_full: fa.EdgeKField[float],
    ddxt_z_full: fa.EdgeKField[float],
    vt: fa.EdgeKField[float],
    z_w_concorr_me: fa.EdgeKField[float],
    wgtfac_e: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    z_vt_ie: fa.EdgeKField[float],
    z_kin_hor_e: fa.EdgeKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
) -> tuple[
    fa.EdgeKField[float],
    fa.EdgeKField[float],
    fa.EdgeKField[float],
    fa.EdgeKField[float],
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_stencils_35_36(
    vn: fa.EdgeKField[float],
    ddxn_z_full: fa.EdgeKField[float],
    ddxt_z_full: fa.EdgeKField[float],
    vt: fa.EdgeKField[float],
    z_w_concorr_me: fa.EdgeKField[float],
    wgtfac_e: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    z_vt_ie: fa.EdgeKField[float],
    z_kin_hor_e: fa.EdgeKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_stencils_37_38(
    vn: fa.EdgeKField[float],
    vt: fa.EdgeKField[float],
    vn_ie: fa.EdgeKField[float],
    z_vt_ie: fa.EdgeKField[float],
    z_kin_hor_e: fa.EdgeKField[float],
    wgtfacq_e_dsl: fa.EdgeKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_horizontal_kinetic_energy(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_start + 1),
        },
    )
    _extrapolate_at_top(
        vn,
        wgtfacq_e_dsl,
        out=vn_ie,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.field_operator
def _stencils_39_40(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    z_w_concorr_me: fa.EdgeKField[float],
    wgtfac_c: fa.CellKField[float],
    wgtfacq_c_dsl: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex_plus1: gtx.int32,
    nlev: gtx.int32,
) -> fa.CellKField[float]:
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_39_40(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], float],
    z_w_concorr_me: fa.EdgeKField[float],
    wgtfac_c: fa.CellKField[float],
    wgtfacq_c_dsl: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    nflatlev_startindex_plus1: gtx.int32,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _stencils_42_44_45(
    z_w_expl: fa.CellKField[float],
    w_nnow: fa.CellKField[float],
    ddt_w_adv_ntl1: fa.CellKField[float],
    ddt_w_adv_ntl2: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    z_contr_w_fl_l: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    z_beta: fa.CellKField[float],
    exner_nnow: fa.CellKField[float],
    rho_nnow: fa.CellKField[float],
    theta_v_nnow: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    z_q: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
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
    z_q = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_q)

    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_42_44_45_45b(
    z_w_expl: fa.CellKField[float],
    w_nnow: fa.CellKField[float],
    ddt_w_adv_ntl1: fa.CellKField[float],
    ddt_w_adv_ntl2: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    z_contr_w_fl_l: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    z_beta: fa.CellKField[float],
    exner_nnow: fa.CellKField[float],
    rho_nnow: fa.CellKField[float],
    theta_v_nnow: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    z_q: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _stencils_42_44_45(
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _init_cell_kdim_field_with_zero_vp(
        out=z_alpha,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.field_operator
def _stencils_43_44_45(
    z_w_expl: fa.CellKField[float],
    w_nnow: fa.CellKField[float],
    ddt_w_adv_ntl1: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    z_contr_w_fl_l: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    z_beta: fa.CellKField[float],
    exner_nnow: fa.CellKField[float],
    rho_nnow: fa.CellKField[float],
    theta_v_nnow: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    z_q: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
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
    z_q = where(k_field == 0, _init_cell_kdim_field_with_zero_vp(), z_q)

    return z_w_expl, z_contr_w_fl_l, z_beta, z_alpha, z_q


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_43_44_45_45b(
    z_w_expl: fa.CellKField[float],
    w_nnow: fa.CellKField[float],
    ddt_w_adv_ntl1: fa.CellKField[float],
    z_th_ddz_exner_c: fa.CellKField[float],
    z_contr_w_fl_l: fa.CellKField[float],
    rho_ic: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    vwind_expl_wgt: fa.CellField[float],
    z_beta: fa.CellKField[float],
    exner_nnow: fa.CellKField[float],
    rho_nnow: fa.CellKField[float],
    theta_v_nnow: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_alpha: fa.CellKField[float],
    vwind_impl_wgt: fa.CellField[float],
    theta_v_ic: fa.CellKField[float],
    z_q: fa.CellKField[float],
    k_field: fa.KField[gtx.int32],
    rd: float,
    cvd: float,
    dtime: float,
    cpd: float,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _stencils_43_44_45(
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _init_cell_kdim_field_with_zero_vp(
        out=z_alpha,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_47_48_49(
    w_nnew: fa.CellKField[float],
    z_contr_w_fl_l: fa.CellKField[float],
    w_concorr_c: fa.CellKField[float],
    z_rho_expl: fa.CellKField[float],
    z_exner_expl: fa.CellKField[float],
    rho_nnow: fa.CellKField[float],
    inv_ddqz_z_full: fa.CellKField[float],
    z_flxdiv_mass: fa.CellKField[float],
    exner_pr: fa.CellKField[float],
    z_beta: fa.CellKField[float],
    z_flxdiv_theta: fa.CellKField[float],
    theta_v_ic: fa.CellKField[float],
    ddt_exner_phy: fa.CellKField[float],
    dtime: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_end: gtx.int32,
    vertical_start: gtx.int32,
):
    _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c,
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
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
        out=(z_rho_expl, z_exner_expl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_61_62(
    rho_now: fa.CellKField[float],
    grf_tend_rho: fa.CellKField[float],
    theta_v_now: fa.CellKField[float],
    grf_tend_thv: fa.CellKField[float],
    w_now: fa.CellKField[float],
    grf_tend_w: fa.CellKField[float],
    rho_new: fa.CellKField[float],
    exner_new: fa.CellKField[float],
    w_new: fa.CellKField[float],
    dtime: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_density_exner_wind(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=(rho_new, exner_new, w_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _update_wind(
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
