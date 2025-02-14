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

import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    _add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    _add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    _add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    _apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    _apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.apply_weighted_2nd_and_4th_order_divergence_damping import (
    _apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import _compute_graddiv2_of_vn
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    _compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_edge_kdim_fields_with_zero_wp import (
    _init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.mo_math_gradients_grad_green_gauss_cell_dsl import (
    _mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28_predictor(
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[vpfloat],
    pos_on_tplane_e_1: gtx.Field[[dims.ECDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    rho_ref_me: fa.EdgeKField[vpfloat],
    theta_ref_me: fa.EdgeKField[vpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.ECDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_hydro_corr: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    z_rho_e: fa.EdgeKField[wpfloat],
    horz_idx: fa.EdgeField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    grav_o_cpd: wpfloat,
    p_dthalf: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    limited_area: bool,
    start_edge_halo_level_2: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    end_edge_halo: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_end: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    MIURA: gtx.int32,
    TAYLOR_HYDRO: gtx.int32,
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    vert_idx = broadcast(vert_idx, (EdgeDim, KDim))

    (
        z_grad_rth_1,
        z_grad_rth_2,
        z_grad_rth_3,
        z_grad_rth_4,
    ) = _mo_math_gradients_grad_green_gauss_cell_dsl(
        p_ccpr1=z_rth_pr_1,
        p_ccpr2=z_rth_pr_2,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
    )  # if (iadv_rhotheta == MIURA) else (z_grad_rth_1, z_grad_rth_2, z_grad_rth_3, z_grad_rth_4,)

    (z_rho_e, z_theta_v_e) = (
        where(
            (start_edge_halo_level_2 <= horz_idx < end_edge_halo_level_2),
            _init_two_edge_kdim_fields_with_zero_wp(),
            (z_rho_e, z_theta_v_e),
        )
        if iadv_rhotheta <= 2
        else (z_rho_e, z_theta_v_e)
    )

    (z_rho_e, z_theta_v_e) = (
        where(
            (start_edge_lateral_boundary <= horz_idx < end_edge_halo),
            _init_two_edge_kdim_fields_with_zero_wp(),
            (z_rho_e, z_theta_v_e),
        )
        if limited_area & (iadv_rhotheta <= 2)
        else (z_rho_e, z_theta_v_e)
    )

    (z_rho_e, z_theta_v_e) = (
        where(
            (start_edge_lateral_boundary_level_7 <= horz_idx < end_edge_halo),
            _compute_horizontal_advection_of_rho_and_theta(
                p_vn=p_vn,
                p_vt=p_vt,
                pos_on_tplane_e_1=pos_on_tplane_e_1,
                pos_on_tplane_e_2=pos_on_tplane_e_2,
                primal_normal_cell_1=primal_normal_cell_1,
                dual_normal_cell_1=dual_normal_cell_1,
                primal_normal_cell_2=primal_normal_cell_2,
                dual_normal_cell_2=dual_normal_cell_2,
                p_dthalf=p_dthalf,
                rho_ref_me=rho_ref_me,
                theta_ref_me=theta_ref_me,
                z_grad_rth_1=z_grad_rth_1,
                z_grad_rth_2=z_grad_rth_2,
                z_grad_rth_3=z_grad_rth_3,
                z_grad_rth_4=z_grad_rth_4,
                z_rth_pr_1=z_rth_pr_1,
                z_rth_pr_2=z_rth_pr_2,
            ),
            (z_rho_e, z_theta_v_e),
        )
        if (iadv_rhotheta == MIURA) & (iadv_rhotheta <= 2)
        else (z_rho_e, z_theta_v_e)
    )

    z_gradh_exner = where(
        (start_edge_nudging_level_2 <= horz_idx < end_edge_local) & (vert_idx < nflatlev),
        _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=inv_dual_edge_length, z_exner_ex_pr=z_exner_ex_pr
        ),
        z_gradh_exner,
    )

    z_gradh_exner = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local)
            & (nflatlev < vert_idx < (nflat_gradp + int32(1))),
            _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
                inv_dual_edge_length=inv_dual_edge_length,
                z_exner_ex_pr=z_exner_ex_pr,
                ddxn_z_full=ddxn_z_full,
                c_lin_e=c_lin_e,
                z_dexner_dz_c_1=z_dexner_dz_c_1,
            ),
            z_gradh_exner,
        )
        if igradp_method == TAYLOR_HYDRO
        else z_gradh_exner
    )

    z_gradh_exner = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local)
            & (nflat_gradp + int32(1) <= vert_idx),
            _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
                inv_dual_edge_length=inv_dual_edge_length,
                z_exner_ex_pr=z_exner_ex_pr,
                zdiff_gradp=zdiff_gradp,
                ikoffset=ikoffset,
                z_dexner_dz_c_1=z_dexner_dz_c_1,
                z_dexner_dz_c_2=z_dexner_dz_c_2,
            ),
            z_gradh_exner,
        )
        if igradp_method == TAYLOR_HYDRO
        else z_gradh_exner
    )

    z_hydro_corr = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local) & (nlev - 1 <= vert_idx),
            _compute_hydrostatic_correction_term(
                theta_v=theta_v,
                ikoffset=ikoffset,
                zdiff_gradp=zdiff_gradp,
                theta_v_ic=theta_v_ic,
                inv_ddqz_z_full=inv_ddqz_z_full,
                inv_dual_edge_length=inv_dual_edge_length,
                grav_o_cpd=grav_o_cpd,
            ),
            z_hydro_corr,
        )
        if igradp_method == TAYLOR_HYDRO
        else z_hydro_corr
    )

    hydro_corr_horizontal = where((vert_idx == nlev - 1), z_hydro_corr, 0.0)

    z_gradh_exner = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_end),
            _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
                ipeidx_dsl=ipeidx_dsl,
                pg_exdist=pg_exdist,
                z_hydro_corr=hydro_corr_horizontal,
                z_gradh_exner=z_gradh_exner,
            ),
            z_gradh_exner,
        )
        if igradp_method == TAYLOR_HYDRO
        else z_gradh_exner
    )

    vn = where(
        (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
        _add_temporal_tendencies_to_vn(
            vn_nnow=vn,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            dtime=dtime,
            cpd=cpd,
        ),
        vn,
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _add_analysis_increments_to_vn(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
            vn,
        )
        if is_iau_active
        else vn
    )

    return z_rho_e, z_theta_v_e, z_gradh_exner, vn


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28_corrector(
    hmask_dd3d: fa.EdgeField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddt_vn_apc_ntl2: fa.EdgeKField[vpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    scal_divdamp: fa.KField[wpfloat],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    horz_idx: fa.EdgeField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    divdamp_fac: wpfloat,
    divdamp_fac_o2: wpfloat,
    divdamp_order: gtx.int32,
    scal_divdamp_o2: wpfloat,
    limited_area: bool,
    itime_scheme: gtx.int32,
    COMBINED: gtx.int32,
    FOURTH_ORDER: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    kstart_dd3d: gtx.int32,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    vert_idx = broadcast(vert_idx, (EdgeDim, KDim))

    z_graddiv_vn = where(
        (start_edge_lateral_boundary_level_7 <= horz_idx < end_edge_halo_level_2)
        & (kstart_dd3d <= vert_idx),
        _add_vertical_wind_derivative_to_divergence_damping(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
        ),
        z_graddiv_vn,
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
                vn_nnow=vn,
                ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
                ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
                ddt_vn_phy=ddt_vn_phy,
                z_theta_v_e=z_theta_v_e,
                z_gradh_exner=z_gradh_exner,
                dtime=dtime,
                wgt_nnow_vel=wgt_nnow_vel,
                wgt_nnew_vel=wgt_nnew_vel,
                cpd=cpd,
            ),
            vn,
        )
        if itime_scheme == 4
        else vn
    )

    z_graddiv2_vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _compute_graddiv2_of_vn(geofac_grdiv=geofac_grdiv, z_graddiv_vn=z_graddiv_vn),
            z_graddiv2_vn,
        )
        if (divdamp_order == COMBINED) | (divdamp_order == FOURTH_ORDER)
        else z_graddiv2_vn
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _apply_2nd_order_divergence_damping(
                z_graddiv_vn=z_graddiv_vn, vn=vn, scal_divdamp_o2=scal_divdamp_o2
            ),
            vn,
        )
        if ((divdamp_order == COMBINED) & (scal_divdamp_o2 > 1.0e-6))
        else vn
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _apply_weighted_2nd_and_4th_order_divergence_damping(
                scal_divdamp=scal_divdamp,
                bdy_divdamp=bdy_divdamp,
                nudgecoeff_e=nudgecoeff_e,
                z_graddiv2_vn=z_graddiv2_vn,
                vn=vn,
            ),
            vn,
        )
        if ((divdamp_order == COMBINED) & (divdamp_fac_o2 <= (4.0 * divdamp_fac)) & limited_area)
        else vn
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _apply_4th_order_divergence_damping(
                scal_divdamp=scal_divdamp, z_graddiv2_vn=z_graddiv2_vn, vn=vn
            ),
            vn,
        )
        if (
            (divdamp_order == COMBINED)
            & (divdamp_fac_o2 <= (4.0 * divdamp_fac))
            & (not limited_area)
        )
        else vn
    )

    vn = (
        where(
            (start_edge_nudging_level_2 <= horz_idx < end_edge_local),
            _add_analysis_increments_to_vn(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
            vn,
        )
        if is_iau_active
        else vn
    )

    return z_graddiv_vn, vn


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28(
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[vpfloat],
    pos_on_tplane_e_1: gtx.Field[[dims.ECDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    rho_ref_me: fa.EdgeKField[vpfloat],
    theta_ref_me: fa.EdgeKField[vpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.ECDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[vpfloat],
    hmask_dd3d: fa.EdgeField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddt_vn_apc_ntl2: fa.EdgeKField[vpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    vn_incr: fa.EdgeKField[vpfloat],
    horz_idx: fa.EdgeField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_hydro_corr: fa.EdgeKField[vpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    scal_divdamp: fa.KField[wpfloat],
    z_rho_e: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    divdamp_fac: wpfloat,
    divdamp_fac_o2: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    itime_scheme: gtx.int32,
    p_dthalf: wpfloat,
    grav_o_cpd: wpfloat,
    divdamp_order: gtx.int32,
    scal_divdamp_o2: wpfloat,
    istep: gtx.int32,
    limited_area: bool,
    start_edge_halo_level_2: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    end_edge_halo: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_end: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    MIURA: gtx.int32,
    TAYLOR_HYDRO: gtx.int32,
    nlev: gtx.int32,
    kstart_dd3d: gtx.int32,
    COMBINED: gtx.int32,
    FOURTH_ORDER: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    (
        z_rho_e,
        z_theta_v_e,
        z_gradh_exner,
        vn,
    ) = (
        _fused_solve_nonhydro_stencil_15_to_28_predictor(
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            z_rth_pr_1=z_rth_pr_1,
            z_rth_pr_2=z_rth_pr_2,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_exner_ex_pr=z_exner_ex_pr,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
            z_gradh_exner=z_gradh_exner,
            z_hydro_corr=z_hydro_corr,
            z_rho_e=z_rho_e,
            z_theta_v_e=z_theta_v_e,
            theta_v=theta_v,
            ikoffset=ikoffset,
            zdiff_gradp=zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            inv_dual_edge_length=inv_dual_edge_length,
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            vn_incr=vn_incr,
            vn=vn,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            grav_o_cpd=grav_o_cpd,
            p_dthalf=p_dthalf,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            limited_area=limited_area,
            start_edge_halo_level_2=start_edge_halo_level_2,
            end_edge_halo_level_2=end_edge_halo_level_2,
            start_edge_lateral_boundary=start_edge_lateral_boundary,
            end_edge_halo=end_edge_halo,
            start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
            start_edge_nudging_level_2=start_edge_nudging_level_2,
            end_edge_local=end_edge_local,
            end_edge_end=end_edge_end,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
            MIURA=MIURA,
            TAYLOR_HYDRO=TAYLOR_HYDRO,
            nlev=nlev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
        if istep == 1
        else (z_rho_e, z_theta_v_e, z_gradh_exner, vn)
    )

    (z_graddiv_vn, vn) = (
        _fused_solve_nonhydro_stencil_15_to_28_corrector(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            z_dwdz_dd=z_dwdz_dd,
            inv_dual_edge_length=inv_dual_edge_length,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_graddiv_vn=z_graddiv_vn,
            vn_incr=vn_incr,
            vn=vn,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            z_graddiv2_vn=z_graddiv2_vn,
            geofac_grdiv=geofac_grdiv,
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            divdamp_fac=divdamp_fac,
            divdamp_fac_o2=divdamp_fac_o2,
            divdamp_order=divdamp_order,
            scal_divdamp_o2=scal_divdamp_o2,
            limited_area=limited_area,
            itime_scheme=itime_scheme,
            COMBINED=COMBINED,
            FOURTH_ORDER=FOURTH_ORDER,
            start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
            end_edge_halo_level_2=end_edge_halo_level_2,
            start_edge_nudging_level_2=start_edge_nudging_level_2,
            end_edge_local=end_edge_local,
            kstart_dd3d=kstart_dd3d,
        )
        if istep > 1
        else (z_graddiv_vn, vn)
    )

    return z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn


@gtx.program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_15_to_28(
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[vpfloat],
    pos_on_tplane_e_1: gtx.Field[[dims.ECDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_1: gtx.Field[[dims.ECDim], wpfloat],
    primal_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    dual_normal_cell_2: gtx.Field[[dims.ECDim], wpfloat],
    rho_ref_me: fa.EdgeKField[vpfloat],
    theta_ref_me: fa.EdgeKField[vpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[[dims.ECDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.ECDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[vpfloat],
    hmask_dd3d: fa.EdgeField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddt_vn_apc_ntl2: fa.EdgeKField[vpfloat],
    ddt_vn_apc_ntl1: fa.EdgeKField[vpfloat],
    ddt_vn_phy: fa.EdgeKField[vpfloat],
    vn_incr: fa.EdgeKField[vpfloat],
    horz_idx: fa.EdgeField[gtx.int32],
    vert_idx: fa.KField[gtx.int32],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_hydro_corr: fa.EdgeKField[vpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    scal_divdamp: fa.KField[wpfloat],
    z_rho_e: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    divdamp_fac: wpfloat,
    divdamp_fac_o2: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    itime_scheme: gtx.int32,
    p_dthalf: wpfloat,
    grav_o_cpd: wpfloat,
    limited_area: bool,
    divdamp_order: gtx.int32,
    scal_divdamp_o2: wpfloat,
    istep: gtx.int32,
    start_edge_halo_level_2: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    end_edge_halo: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_end: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    MIURA: gtx.int32,
    TAYLOR_HYDRO: gtx.int32,
    nlev: gtx.int32,
    kstart_dd3d: gtx.int32,
    COMBINED: gtx.int32,
    FOURTH_ORDER: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
):
    _fused_solve_nonhydro_stencil_15_to_28(
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        p_vn=p_vn,
        p_vt=p_vt,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        ddxn_z_full=ddxn_z_full,
        c_lin_e=c_lin_e,
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v=theta_v,
        ikoffset=ikoffset,
        zdiff_gradp=zdiff_gradp,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        z_dwdz_dd=z_dwdz_dd,
        inv_dual_edge_length=inv_dual_edge_length,
        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_phy=ddt_vn_phy,
        vn_incr=vn_incr,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        bdy_divdamp=bdy_divdamp,
        nudgecoeff_e=nudgecoeff_e,
        z_hydro_corr=z_hydro_corr,
        geofac_grdiv=geofac_grdiv,
        z_graddiv2_vn=z_graddiv2_vn,
        scal_divdamp=scal_divdamp,
        z_rho_e=z_rho_e,
        z_theta_v_e=z_theta_v_e,
        z_gradh_exner=z_gradh_exner,
        vn=vn,
        z_graddiv_vn=z_graddiv_vn,
        divdamp_fac=divdamp_fac,
        divdamp_fac_o2=divdamp_fac_o2,
        wgt_nnow_vel=wgt_nnow_vel,
        wgt_nnew_vel=wgt_nnew_vel,
        dtime=dtime,
        cpd=cpd,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        itime_scheme=itime_scheme,
        p_dthalf=p_dthalf,
        grav_o_cpd=grav_o_cpd,
        divdamp_order=divdamp_order,
        scal_divdamp_o2=scal_divdamp_o2,
        istep=istep,
        limited_area=limited_area,
        start_edge_halo_level_2=start_edge_halo_level_2,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        end_edge_halo=end_edge_halo,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        end_edge_end=end_edge_end,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        MIURA=MIURA,
        TAYLOR_HYDRO=TAYLOR_HYDRO,
        nlev=nlev,
        kstart_dd3d=kstart_dd3d,
        COMBINED=COMBINED,
        FOURTH_ORDER=FOURTH_ORDER,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        out=(z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn),
    )
