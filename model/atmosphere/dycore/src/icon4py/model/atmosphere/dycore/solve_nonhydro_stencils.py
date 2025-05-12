# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where

from icon4py.model.atmosphere.dycore.dycore_utils import (
    _broadcast_zero_to_three_edge_kdim_fields_wp,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers,
)
from icon4py.model.atmosphere.dycore.stencils.compute_virtual_potential_temperatures_and_pressure_gradient import (
    _compute_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_vp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges import (
    _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    _update_density_exner_wind,
)
from icon4py.model.atmosphere.dycore.stencils.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa


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
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
]:
    (z_rth_pr_1, z_rth_pr_2) = concat_where(
        dims.KDim == 0,
        _compute_perturbation_of_rho_and_theta(rho, rho_ref_mc, theta_v, theta_ref_mc),
        (z_rth_pr_1, z_rth_pr_2),
    )

    (rho_ic, z_rth_pr_1, z_rth_pr_2) = concat_where(
        dims.KDim >= 1,
        _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
            wgtfac_c, rho, rho_ref_mc, theta_v, theta_ref_mc
        ),
        (rho_ic, z_rth_pr_1, z_rth_pr_2),
    )

    (z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c) = concat_where(
        dims.KDim >= 1,
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
    z_w_concorr_me = concat_where(
        dims.KDim >= nflatlev_startindex,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, vt),
        z_w_concorr_me,
    )
    (vn_ie, z_vt_ie, z_kin_hor_e) = concat_where(
        dims.KDim >= 1,
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
    w_concorr_c = concat_where(
        dims.KDim >= nflatlev_startindex_plus1,
        _compute_contravariant_correction_of_w(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        w_concorr_c,
    )

    w_concorr_c = concat_where(
        dims.KDim == nlev,
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
