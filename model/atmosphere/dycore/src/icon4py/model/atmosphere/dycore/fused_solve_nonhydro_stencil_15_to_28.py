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
from gt4py.next.ffront.fbuiltins import broadcast, int32, where
from gt4py.next.common import GridType

from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    _mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.atmosphere.dycore.apply_4th_order_divergence_damping import (
    _apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.add_vertical_wind_derivative_to_divergence_damping import (
    _add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_extner_pressure_for_multiple_levels import (
    _compute_horizontal_gradient_of_extner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.compute_hydrostatic_correction_term import (
    _compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn import (
    _add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.compute_graddiv2_of_vn import (
    _compute_graddiv2_of_vn,
)
from icon4py.model.atmosphere.dycore.apply_2nd_order_divergence_damping import (
    _apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.apply_weighted_2nd_and_4th_order_divergence_damping import (
    _apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.add_analysis_increments_to_vn import (
    _add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import _set_zero_e_k
from icon4py.model.common.dimension import (
    C2E2CODim,
    CellDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28_predictor(
    geofac_grg_x: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    p_vn: gtx.Field[[EdgeDim, KDim], wpfloat],
    p_vt: gtx.Field[[EdgeDim, KDim], vpfloat],
    pos_on_tplane_e_1: gtx.Field[[ECDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[[ECDim], wpfloat],
    primal_normal_cell_1: gtx.Field[[ECDim], wpfloat],
    dual_normal_cell_1: gtx.Field[[ECDim], wpfloat],
    primal_normal_cell_2: gtx.Field[[ECDim], wpfloat],
    dual_normal_cell_2: gtx.Field[[ECDim], wpfloat],
    rho_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
    theta_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
    z_rth_pr_1: gtx.Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: gtx.Field[[CellDim, KDim], vpfloat],
    ddxn_z_full: gtx.Field[[EdgeDim, KDim], vpfloat],
    c_lin_e: gtx.Field[[EdgeDim, E2CDim], wpfloat],
    z_exner_ex_pr: gtx.Field[[CellDim, KDim], vpfloat],
    z_dexner_dz_c_1: gtx.Field[[CellDim, KDim], vpfloat],
    z_dexner_dz_c_2: gtx.Field[[CellDim, KDim], vpfloat],
    theta_v: gtx.Field[[CellDim, KDim], wpfloat],
    ikoffset: gtx.Field[[ECDim, KDim], int32],
    zdiff_gradp: gtx.Field[[ECDim, KDim], vpfloat],
    theta_v_ic: gtx.Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: gtx.Field[[CellDim, KDim], vpfloat],
    ipeidx_dsl: gtx.Field[[EdgeDim, KDim], bool],
    pg_exdist: gtx.Field[[EdgeDim, KDim], vpfloat],
    inv_dual_edge_length: gtx.Field[[EdgeDim], wpfloat],
    vn_nnow: gtx.Field[[EdgeDim, KDim], wpfloat],
    ddt_vn_apc_ntl1: gtx.Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_phy: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn_incr: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_e: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_hydro_corr: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_gradh_exner: gtx.Field[[EdgeDim, KDim], vpfloat],
    z_rho_e: gtx.Field[[EdgeDim, KDim], wpfloat],
    horz_idx: gtx.Field[[EdgeDim], int32],
    vert_idx: gtx.Field[[KDim], int32],
    grav_o_cpd: wpfloat,
    p_dthalf: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    limited_area: bool,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
    horizontal_lower_00: int32,
    horizontal_upper_00: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_1: int32,
    horizontal_upper_1: int32,
    horizontal_lower_3: int32,
    horizontal_upper_3: int32,
    horizontal_lower_4: int32,
    horizontal_upper_4: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
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
    )

    # (z_rho_e, z_theta_v_e) = where(
    #     (horizontal_lower_01 <= horz_idx < horizontal_upper_01),
    #     (_set_zero_e_k(), _set_zero_e_k()),
    #     (z_rho_e, z_theta_v_e),
    # )

    z_rho_e = where(
        (horizontal_lower_01 <= horz_idx < horizontal_upper_01),
        _set_zero_e_k(),
        z_rho_e,
    )

    z_theta_v_e = where(
        (horizontal_lower_01 <= horz_idx < horizontal_upper_01),
        _set_zero_e_k(),
        z_theta_v_e,
    )

    # (z_rho_e, z_theta_v_e) = where(
    #     (horizontal_lower_4 <= horz_idx < horizontal_upper_4),
    #     (_set_zero_e_k(), _set_zero_e_k()),
    #     (z_rho_e, z_theta_v_e),
    # ) if limited_area else (z_rho_e, z_theta_v_e)

    z_rho_e = where(
        (horizontal_lower_4 <= horz_idx < horizontal_upper_4),
        _set_zero_e_k(),
        z_rho_e,
    ) if limited_area else z_rho_e

    z_theta_v_e = where(
        (horizontal_lower_4 <= horz_idx < horizontal_upper_4),
        _set_zero_e_k(),
        z_theta_v_e,
    ) if limited_area else z_theta_v_e

    (z_rho_e, z_theta_v_e) = where(
        (horizontal_lower_1 <= horz_idx < horizontal_upper_1),
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

    z_gradh_exner = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0) & (vert_idx < nflatlev),
        _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=inv_dual_edge_length, z_exner_ex_pr=z_exner_ex_pr
        ),
        z_gradh_exner,
    )

    z_gradh_exner = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0)
        & (nflatlev < vert_idx < nflat_gradp + int32(1)),
        _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
        ),
        z_gradh_exner,
    )

    z_gradh_exner = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0)
        & (nflat_gradp + int32(1) <= vert_idx),
        _compute_horizontal_gradient_of_extner_pressure_for_multiple_levels(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=z_exner_ex_pr,
            zdiff_gradp=zdiff_gradp,
            ikoffset=ikoffset,
            z_dexner_dz_c_1=z_dexner_dz_c_1,
            z_dexner_dz_c_2=z_dexner_dz_c_2,
        ),
        z_gradh_exner,
    )

    z_hydro_corr = _compute_hydrostatic_correction_term(
        theta_v=theta_v,
        ikoffset=ikoffset,
        zdiff_gradp=zdiff_gradp,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        inv_dual_edge_length=inv_dual_edge_length,
        grav_o_cpd=grav_o_cpd,
    )

    z_gradh_exner = where(
        (horizontal_lower_3 <= horz_idx < horizontal_upper_3),
        _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            z_hydro_corr=z_hydro_corr,
            z_gradh_exner=z_gradh_exner,
        ),
        z_gradh_exner,
    )

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _add_temporal_tendencies_to_vn(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            dtime=dtime,
            cpd=cpd,
        ),
        vn,
    )

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _add_analysis_increments_to_vn(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
        vn,
    ) if is_iau_active else vn

    return z_rho_e, z_theta_v_e, z_gradh_exner, vn


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28_corrector(
    hmask_dd3d: gtx.Field[[EdgeDim], wpfloat],
    scalfac_dd3d: gtx.Field[[KDim], wpfloat],
    z_dwdz_dd: gtx.Field[[CellDim, KDim], vpfloat],
    inv_dual_edge_length: gtx.Field[[EdgeDim], wpfloat],
    ddt_vn_apc_ntl2: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn_nnow: gtx.Field[[EdgeDim, KDim], wpfloat],
    ddt_vn_apc_ntl1: gtx.Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_phy: gtx.Field[[EdgeDim, KDim], vpfloat],
    z_graddiv_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn_incr: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_e: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_gradh_exner: gtx.Field[[EdgeDim, KDim], vpfloat],
    z_graddiv2_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
    geofac_grdiv: gtx.Field[[EdgeDim, E2C2EODim], wpfloat],
    scal_divdamp: gtx.Field[[KDim], wpfloat],
    bdy_divdamp: gtx.Field[[KDim], wpfloat],
    nudgecoeff_e: gtx.Field[[EdgeDim], wpfloat],
    horz_idx: gtx.Field[[EdgeDim], int32],
    vert_idx: gtx.Field[[KDim], int32],
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    lhdiff_rcf: bool,
    divdamp_fac: wpfloat,
    divdamp_fac_o2: wpfloat,
    divdamp_order: int32,
    scal_divdamp_o2: wpfloat,
    limited_area: bool,
    itime_scheme: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
    horizontal_lower_2: int32,
    horizontal_upper_2: int32,
    kstart_dd3d: int32,
) -> tuple[gtx.Field[[EdgeDim, KDim], wpfloat], gtx.Field[[EdgeDim, KDim], wpfloat]]:
    vert_idx = broadcast(vert_idx, (EdgeDim, KDim))

    z_graddiv_vn = where(
        (horizontal_lower_2 <= horz_idx < horizontal_upper_2) & (kstart_dd3d <= vert_idx),
        _add_vertical_wind_derivative_to_divergence_damping(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
        ),
        z_graddiv_vn,
    )

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
            vn_nnow=vn_nnow,
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
    ) if itime_scheme == 4 else vn

    z_graddiv2_vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _compute_graddiv2_of_vn(geofac_grdiv=geofac_grdiv, z_graddiv_vn=z_graddiv_vn),
        z_graddiv2_vn,
    )

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _apply_2nd_order_divergence_damping(
            z_graddiv_vn=z_graddiv_vn, vn=vn, scal_divdamp_o2=scal_divdamp_o2
        ),
        vn,
    ) if (lhdiff_rcf & (divdamp_order == int32(24)) & (scal_divdamp_o2 > 1.0e-6)) else vn

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _apply_weighted_2nd_and_4th_order_divergence_damping(
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
        ),
        vn,
    ) if ((divdamp_order == int32(24)) & (divdamp_fac_o2 <= (4.0 * divdamp_fac)) & limited_area) else vn

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _apply_4th_order_divergence_damping(
            scal_divdamp=scal_divdamp, z_graddiv2_vn=z_graddiv2_vn, vn=vn
        ),
        vn,
    ) if ((divdamp_order == int32(24)) & (divdamp_fac_o2 <= (4.0 * divdamp_fac))) else vn

    vn = where(
        (horizontal_lower_0 <= horz_idx < horizontal_upper_0),
        _add_analysis_increments_to_vn(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
        vn,
    ) if is_iau_active else vn

    return z_graddiv_vn, vn


@gtx.field_operator
def _fused_solve_nonhydro_stencil_15_to_28(
    geofac_grg_x: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[CellDim, C2E2CODim], wpfloat],
    p_vn: gtx.Field[[EdgeDim, KDim], wpfloat],
    p_vt: gtx.Field[[EdgeDim, KDim], vpfloat],
    pos_on_tplane_e_1: gtx.Field[[ECDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[[ECDim], wpfloat],
    primal_normal_cell_1: gtx.Field[[ECDim], wpfloat],
    dual_normal_cell_1: gtx.Field[[ECDim], wpfloat],
    primal_normal_cell_2: gtx.Field[[ECDim], wpfloat],
    dual_normal_cell_2: gtx.Field[[ECDim], wpfloat],
    rho_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
    theta_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
    z_rth_pr_1: gtx.Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: gtx.Field[[CellDim, KDim], vpfloat],
    ddxn_z_full: gtx.Field[[EdgeDim, KDim], vpfloat],
    c_lin_e: gtx.Field[[EdgeDim, E2CDim], wpfloat],
    z_exner_ex_pr: gtx.Field[[CellDim, KDim], vpfloat],
    z_dexner_dz_c_1: gtx.Field[[CellDim, KDim], vpfloat],
    z_dexner_dz_c_2: gtx.Field[[CellDim, KDim], vpfloat],
    theta_v: gtx.Field[[CellDim, KDim], wpfloat],
    ikoffset: gtx.Field[[ECDim, KDim], int32],
    zdiff_gradp: gtx.Field[[ECDim, KDim], vpfloat],
    theta_v_ic: gtx.Field[[CellDim, KDim], wpfloat],
    inv_ddqz_z_full: gtx.Field[[CellDim, KDim], vpfloat],
    ipeidx_dsl: gtx.Field[[EdgeDim, KDim], bool],
    pg_exdist: gtx.Field[[EdgeDim, KDim], vpfloat],
    hmask_dd3d: gtx.Field[[EdgeDim], wpfloat],
    scalfac_dd3d: gtx.Field[[KDim], wpfloat],
    z_dwdz_dd: gtx.Field[[CellDim, KDim], vpfloat],
    inv_dual_edge_length: gtx.Field[[EdgeDim], wpfloat],
    ddt_vn_apc_ntl2: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn_nnow: gtx.Field[[EdgeDim, KDim], wpfloat],
    ddt_vn_apc_ntl1: gtx.Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_phy: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn_incr: gtx.Field[[EdgeDim, KDim], vpfloat],
    horz_idx: gtx.Field[[EdgeDim], int32],
    vert_idx: gtx.Field[[KDim], int32],
    bdy_divdamp: gtx.Field[[KDim], wpfloat],
    nudgecoeff_e: gtx.Field[[EdgeDim], wpfloat],
    z_hydro_corr: gtx.Field[[EdgeDim, KDim], vpfloat],
    geofac_grdiv: gtx.Field[[EdgeDim, E2C2EODim], wpfloat],
    z_graddiv2_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
    scal_divdamp: gtx.Field[[KDim], wpfloat],
    z_rho_e: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_theta_v_e: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_gradh_exner: gtx.Field[[EdgeDim, KDim], vpfloat],
    vn: gtx.Field[[EdgeDim, KDim], wpfloat],
    z_graddiv_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
    divdamp_fac: wpfloat,
    divdamp_fac_o2: wpfloat,
    wgt_nnow_vel: wpfloat,
    wgt_nnew_vel: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
    iau_wgt_dyn: wpfloat,
    is_iau_active: bool,
    itime_scheme: int32,
    p_dthalf: wpfloat,
    grav_o_cpd: wpfloat,
    limited_area: bool,
    lhdiff_rcf: bool,
    divdamp_order: int32,
    scal_divdamp_o2: wpfloat,
    istep: int32,
    horizontal_lower_0: int32,
    horizontal_upper_0: int32,
    horizontal_lower_00: int32,
    horizontal_upper_00: int32,
    horizontal_lower_01: int32,
    horizontal_upper_01: int32,
    horizontal_lower_1: int32,
    horizontal_upper_1: int32,
    horizontal_lower_2: int32,
    horizontal_upper_2: int32,
    horizontal_lower_3: int32,
    horizontal_upper_3: int32,
    horizontal_lower_4: int32,
    horizontal_upper_4: int32,
    kstart_dd3d: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat],
    gtx.Field[[EdgeDim, KDim], wpfloat]]:

    (
       z_rho_e,
       z_theta_v_e,
       z_gradh_exner,
       vn,
    ) = _fused_solve_nonhydro_stencil_15_to_28_predictor(
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
       vn_nnow=vn_nnow,
       ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
       ddt_vn_phy=ddt_vn_phy,
       vn_incr=vn_incr,
       vn=vn,
       horz_idx=horz_idx,
       vert_idx=vert_idx,
       grav_o_cpd=grav_o_cpd,
       dtime=dtime,
       cpd=cpd,
       p_dthalf=p_dthalf,
       iau_wgt_dyn=iau_wgt_dyn,
       is_iau_active=is_iau_active,
       limited_area=limited_area,
       horizontal_lower_0=horizontal_lower_0,
       horizontal_upper_0=horizontal_upper_0,
       horizontal_lower_00=horizontal_lower_00,
       horizontal_upper_00=horizontal_upper_00,
       horizontal_lower_01=horizontal_lower_01,
       horizontal_upper_01=horizontal_upper_01,
       horizontal_lower_1=horizontal_lower_1,
       horizontal_upper_1=horizontal_upper_1,
       horizontal_lower_3=horizontal_lower_3,
       horizontal_upper_3=horizontal_upper_3,
       horizontal_lower_4=horizontal_lower_4,
       horizontal_upper_4=horizontal_upper_4,
       nflatlev=nflatlev,
       nflat_gradp=nflat_gradp,
    ) if istep == 1 else (z_rho_e, z_theta_v_e, z_gradh_exner, vn)

    (z_graddiv_vn, vn) = _fused_solve_nonhydro_stencil_15_to_28_corrector(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            z_dwdz_dd=z_dwdz_dd,
            inv_dual_edge_length=inv_dual_edge_length,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            vn_nnow=vn_nnow,
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
            wgt_nnow_vel=wgt_nnew_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            dtime=dtime,
            cpd=cpd,
            iau_wgt_dyn=iau_wgt_dyn,
            is_iau_active=is_iau_active,
            lhdiff_rcf=lhdiff_rcf,
            divdamp_fac=divdamp_fac,
            divdamp_fac_o2=divdamp_fac_o2,
            divdamp_order=divdamp_order,
            scal_divdamp_o2=scal_divdamp_o2,
            limited_area=limited_area,
            itime_scheme=itime_scheme,
            horizontal_lower_0=horizontal_lower_0,
            horizontal_upper_0=horizontal_upper_0,
            horizontal_lower_2=horizontal_lower_2,
            horizontal_upper_2=horizontal_upper_2,
            kstart_dd3d=kstart_dd3d,
        ) if istep > 1 else (z_graddiv_vn, vn)

    return z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn


@gtx.program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_15_to_28(
   geofac_grg_x: gtx.Field[[CellDim, C2E2CODim], wpfloat],
   geofac_grg_y: gtx.Field[[CellDim, C2E2CODim], wpfloat],
   p_vn: gtx.Field[[EdgeDim, KDim], wpfloat],
   p_vt: gtx.Field[[EdgeDim, KDim], vpfloat],
   pos_on_tplane_e_1: gtx.Field[[ECDim], wpfloat],
   pos_on_tplane_e_2: gtx.Field[[ECDim], wpfloat],
   primal_normal_cell_1: gtx.Field[[ECDim], wpfloat],
   dual_normal_cell_1: gtx.Field[[ECDim], wpfloat],
   primal_normal_cell_2: gtx.Field[[ECDim], wpfloat],
   dual_normal_cell_2: gtx.Field[[ECDim], wpfloat],
   rho_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
   theta_ref_me: gtx.Field[[EdgeDim, KDim], vpfloat],
   z_rth_pr_1: gtx.Field[[CellDim, KDim], vpfloat],
   z_rth_pr_2: gtx.Field[[CellDim, KDim], vpfloat],
   ddxn_z_full: gtx.Field[[EdgeDim, KDim], vpfloat],
   c_lin_e: gtx.Field[[EdgeDim, E2CDim], wpfloat],
   z_exner_ex_pr: gtx.Field[[CellDim, KDim], vpfloat],
   z_dexner_dz_c_1: gtx.Field[[CellDim, KDim], vpfloat],
   z_dexner_dz_c_2: gtx.Field[[CellDim, KDim], vpfloat],
   theta_v: gtx.Field[[CellDim, KDim], wpfloat],
   ikoffset: gtx.Field[[ECDim, KDim], int32],
   zdiff_gradp: gtx.Field[[ECDim, KDim], vpfloat],
   theta_v_ic: gtx.Field[[CellDim, KDim], wpfloat],
   inv_ddqz_z_full: gtx.Field[[CellDim, KDim], vpfloat],
   ipeidx_dsl: gtx.Field[[EdgeDim, KDim], bool],
   pg_exdist: gtx.Field[[EdgeDim, KDim], vpfloat],
   hmask_dd3d: gtx.Field[[EdgeDim], wpfloat],
   scalfac_dd3d: gtx.Field[[KDim], wpfloat],
   z_dwdz_dd: gtx.Field[[CellDim, KDim], vpfloat],
   inv_dual_edge_length: gtx.Field[[EdgeDim], wpfloat],
   ddt_vn_apc_ntl2: gtx.Field[[EdgeDim, KDim], vpfloat],
   vn_nnow: gtx.Field[[EdgeDim, KDim], wpfloat],
   ddt_vn_apc_ntl1: gtx.Field[[EdgeDim, KDim], vpfloat],
   ddt_vn_phy: gtx.Field[[EdgeDim, KDim], vpfloat],
   vn_incr: gtx.Field[[EdgeDim, KDim], vpfloat],
   horz_idx: gtx.Field[[EdgeDim], int32],
   vert_idx: gtx.Field[[KDim], int32],
   bdy_divdamp: gtx.Field[[KDim], wpfloat],
   nudgecoeff_e: gtx.Field[[EdgeDim], wpfloat],
   z_hydro_corr: gtx.Field[[EdgeDim, KDim], vpfloat],
   geofac_grdiv: gtx.Field[[EdgeDim, E2C2EODim], wpfloat],
   z_graddiv2_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
   scal_divdamp: gtx.Field[[KDim], wpfloat],
   z_rho_e: gtx.Field[[EdgeDim, KDim], wpfloat],
   z_theta_v_e: gtx.Field[[EdgeDim, KDim], wpfloat],
   z_gradh_exner: gtx.Field[[EdgeDim, KDim], vpfloat],
   vn: gtx.Field[[EdgeDim, KDim], wpfloat],
   z_graddiv_vn: gtx.Field[[EdgeDim, KDim], vpfloat],
   divdamp_fac: wpfloat,
   divdamp_fac_o2: wpfloat,
   wgt_nnow_vel: wpfloat,
   wgt_nnew_vel: wpfloat,
   dtime: wpfloat,
   cpd: wpfloat,
   iau_wgt_dyn: wpfloat,
   is_iau_active: bool,
   itime_scheme: int32,
   p_dthalf: wpfloat,
   grav_o_cpd: wpfloat,
   limited_area: bool,
   lhdiff_rcf: bool,
   divdamp_order: int32,
   scal_divdamp_o2: wpfloat,
   istep: int32,
   horizontal_lower_0: int32,
   horizontal_upper_0: int32,
   horizontal_lower_00: int32,
   horizontal_upper_00: int32,
   horizontal_lower_01: int32,
   horizontal_upper_01: int32,
   horizontal_lower_1: int32,
   horizontal_upper_1: int32,
   horizontal_lower_2: int32,
   horizontal_upper_2: int32,
   horizontal_lower_3: int32,
   horizontal_upper_3: int32,
   horizontal_lower_4: int32,
   horizontal_upper_4: int32,
   kstart_dd3d: int32,
   nflatlev: int32,
   nflat_gradp: int32,
):
    _fused_solve_nonhydro_stencil_15_to_28(
       geofac_grg_x,
       geofac_grg_y,
       p_vn,
       p_vt,
       pos_on_tplane_e_1,
       pos_on_tplane_e_2,
       primal_normal_cell_1,
       dual_normal_cell_1,
       primal_normal_cell_2,
       dual_normal_cell_2,
       rho_ref_me,
       theta_ref_me,
       z_rth_pr_1,
       z_rth_pr_2,
       ddxn_z_full,
       c_lin_e,
       z_exner_ex_pr,
       z_dexner_dz_c_1,
       z_dexner_dz_c_2,
       theta_v,
       ikoffset,
       zdiff_gradp,
       theta_v_ic,
       inv_ddqz_z_full,
       ipeidx_dsl,
       pg_exdist,
       hmask_dd3d,
       scalfac_dd3d,
       z_dwdz_dd,
       inv_dual_edge_length,
       ddt_vn_apc_ntl2,
       vn_nnow,
       ddt_vn_apc_ntl1,
       ddt_vn_phy,
       vn_incr,
       horz_idx,
       vert_idx,
       bdy_divdamp,
       nudgecoeff_e,
       z_hydro_corr,
       geofac_grdiv,
       z_graddiv2_vn,
       scal_divdamp,
        z_rho_e,
       z_theta_v_e,
       z_gradh_exner,
       vn,
       z_graddiv_vn,
       divdamp_fac,
       divdamp_fac_o2,
       wgt_nnow_vel,
       wgt_nnew_vel,
       dtime,
       cpd,
       iau_wgt_dyn,
       is_iau_active,
       itime_scheme,
       p_dthalf,
       grav_o_cpd,
       limited_area,
       lhdiff_rcf,
       divdamp_order,
       scal_divdamp_o2,
       istep,
       horizontal_lower_0,
       horizontal_upper_0,
       horizontal_lower_00,
       horizontal_upper_00,
       horizontal_lower_01,
       horizontal_upper_01,
       horizontal_lower_1,
       horizontal_upper_1,
       horizontal_lower_2,
       horizontal_upper_2,
       horizontal_lower_3,
       horizontal_upper_3,
       horizontal_lower_4,
       horizontal_upper_4,
       kstart_dd3d,
       nflatlev,
       nflat_gradp,
       out=(z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn),
    )
