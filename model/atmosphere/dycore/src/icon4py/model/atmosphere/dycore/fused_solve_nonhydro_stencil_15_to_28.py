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
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where

from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    _mo_math_gradients_grad_green_gauss_cell_dsl,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_4th_order_divdamp import (
    _mo_solve_nonhydro_4th_order_divdamp,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1 import (
    _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_17 import (
    _mo_solve_nonhydro_stencil_17,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_18 import (
    _mo_solve_nonhydro_stencil_18,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_19 import (
    _mo_solve_nonhydro_stencil_19,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_20 import (
    _mo_solve_nonhydro_stencil_20,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_21 import (
    _mo_solve_nonhydro_stencil_21,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_22 import (
    _mo_solve_nonhydro_stencil_22,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_23 import (
    _mo_solve_nonhydro_stencil_23,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_24 import (
    _mo_solve_nonhydro_stencil_24,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_25 import (
    _mo_solve_nonhydro_stencil_25,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_26 import (
    _mo_solve_nonhydro_stencil_26,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_27 import (
    _mo_solve_nonhydro_stencil_27,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_28 import (
    _mo_solve_nonhydro_stencil_28,
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


@field_operator
def _fused_solve_nonhydro_stencil_15_to_28_predictor(
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    vn_incr: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    horz_idx: Field[[EdgeDim], int32],
    vert_idx: Field[[KDim], int32],
    grav_o_cpd: float,
    dtime: float,
    cpd: float,
    p_dthalf: float,
    iau_wgt_dyn: float,
    is_iau_active: bool,
    limited_area: bool,
    idiv_method: int32,
    igradp_method: int32,
    horizontal_lower: int32,
    horizontal_upper: int32,
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
    nlev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
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
    if idiv_method == 1:
        (zero_lower_bound, zero_upper_bound) = (horizontal_lower_01, horizontal_upper_01)
    else:
        (zero_lower_bound, zero_upper_bound) = (horizontal_lower_00, horizontal_upper_00)

    (z_rho_e, z_theta_v_e) = where(
        (zero_lower_bound <= horz_idx < zero_upper_bound),
        (_set_zero_e_k(), _set_zero_e_k()),
        (z_rho_e, z_theta_v_e),
    )
    if limited_area:
        (z_rho_e, z_theta_v_e) = where(
            (horizontal_lower_4 <= horz_idx < horizontal_upper_4),
            (_set_zero_e_k(), _set_zero_e_k()),
            (z_rho_e, z_theta_v_e),
        )

    (z_rho_e, z_theta_v_e) = where(
        (horizontal_lower_1 <= horz_idx < horizontal_upper_1),
        _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1(
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
        (horizontal_lower <= horz_idx < horizontal_upper) & (vert_idx < nflatlev),
        _mo_solve_nonhydro_stencil_18(
            inv_dual_edge_length=inv_dual_edge_length, z_exner_ex_pr=z_exner_ex_pr
        ),
        z_gradh_exner,
    )

    if igradp_method == 3:
        z_gradh_exner = where(
            (horizontal_lower <= horz_idx < horizontal_upper)
            & (nflatlev < vert_idx < nflat_gradp + int32(1)),
            _mo_solve_nonhydro_stencil_19(
                inv_dual_edge_length=inv_dual_edge_length,
                z_exner_ex_pr=z_exner_ex_pr,
                ddxn_z_full=ddxn_z_full,
                c_lin_e=c_lin_e,
                z_dexner_dz_c_1=z_dexner_dz_c_1,
            ),
            z_gradh_exner,
        )

        z_gradh_exner = where(
            (horizontal_lower <= horz_idx < horizontal_upper)
            & (nflat_gradp + int32(1) <= vert_idx),
            _mo_solve_nonhydro_stencil_20(
                inv_dual_edge_length=inv_dual_edge_length,
                z_exner_ex_pr=z_exner_ex_pr,
                zdiff_gradp=zdiff_gradp,
                ikoffset=ikoffset,
                z_dexner_dz_c_1=z_dexner_dz_c_1,
                z_dexner_dz_c_2=z_dexner_dz_c_2,
            ),
            z_gradh_exner,
        )

        z_hydro_corr = _mo_solve_nonhydro_stencil_21(
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
            _mo_solve_nonhydro_stencil_22(
                ipeidx_dsl=ipeidx_dsl,
                pg_exdist=pg_exdist,
                z_hydro_corr=z_hydro_corr,
                z_gradh_exner=z_gradh_exner,
            ),
            z_gradh_exner,
        )

    vn = where(
        (horizontal_lower <= horz_idx < horizontal_upper),
        _mo_solve_nonhydro_stencil_24(
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

    if is_iau_active:
        vn = where(
            (horizontal_lower <= horz_idx < horizontal_upper),
            _mo_solve_nonhydro_stencil_28(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
            vn,
        )

    return z_rho_e, z_theta_v_e, z_gradh_exner, vn


@field_operator
def _fused_solve_nonhydro_stencil_15_to_28_corrector(
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float],
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vn_incr: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    horz_idx: Field[[EdgeDim], int32],
    vert_idx: Field[[KDim], int32],
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    dtime: float,
    cpd: float,
    iau_wgt_dyn: float,
    is_iau_active: bool,
    lhdiff_rcf: bool,
    divdamp_fac: float,
    divdamp_fac_o2: float,
    divdamp_order: int32,
    scal_divdamp_o2: float,
    limited_area: bool,
    itime_scheme: int32,
    horizontal_lower: int32,
    horizontal_upper: int32,
    horizontal_lower_2: int32,
    horizontal_upper_2: int32,
    kstart_dd3d: int32,
    nlev: int32,
) -> tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]:
    vert_idx = broadcast(vert_idx, (EdgeDim, KDim))

    z_graddiv_vn = where(
        (horizontal_lower_2 <= horz_idx < horizontal_upper_2) & (kstart_dd3d <= vert_idx),
        _mo_solve_nonhydro_stencil_17(
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
        ),
        z_graddiv_vn,
    )
    if itime_scheme == 4:
        vn = where(
            (horizontal_lower <= horz_idx < horizontal_upper),
            _mo_solve_nonhydro_stencil_23(
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
        )

    z_graddiv2_vn = where(
        (horizontal_lower <= horz_idx < horizontal_upper),
        _mo_solve_nonhydro_stencil_25(geofac_grdiv=geofac_grdiv, z_graddiv_vn=z_graddiv_vn),
        z_graddiv2_vn,
    )

    if lhdiff_rcf & (divdamp_order == int32(24)) & (scal_divdamp_o2 > 1.0e-6):
        vn = where(
            (horizontal_lower <= horz_idx < horizontal_upper),
            _mo_solve_nonhydro_stencil_26(
                z_graddiv_vn=z_graddiv_vn, vn=vn, scal_divdamp_o2=scal_divdamp_o2
            ),
            vn,
        )
    if (divdamp_order == int32(24)) & (divdamp_fac_o2 <= (4.0 * divdamp_fac)):
        if limited_area:
            vn = where(
                (horizontal_lower <= horz_idx < horizontal_upper),
                _mo_solve_nonhydro_stencil_27(
                    scal_divdamp=scal_divdamp,
                    bdy_divdamp=bdy_divdamp,
                    nudgecoeff_e=nudgecoeff_e,
                    z_graddiv2_vn=z_graddiv2_vn,
                    vn=vn,
                ),
                vn,
            )

        else:
            vn = where(
                (horizontal_lower <= horz_idx < horizontal_upper),
                _mo_solve_nonhydro_4th_order_divdamp(
                    scal_divdamp=scal_divdamp, z_graddiv2_vn=z_graddiv2_vn, vn=vn
                ),
                vn,
            )
    if is_iau_active:
        vn = where(
            (horizontal_lower <= horz_idx < horizontal_upper),
            _mo_solve_nonhydro_stencil_28(vn_incr=vn_incr, vn=vn, iau_wgt_dyn=iau_wgt_dyn),
            vn,
        )

    return z_graddiv_vn, vn


@field_operator
def _fused_solve_nonhydro_stencil_15_to_28(
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float],
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vn_incr: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    horz_idx: Field[[EdgeDim], int32],
    vert_idx: Field[[KDim], int32],
    grav_o_cpd: float,
    p_dthalf: float,
    idiv_method: int32,
    igradp_method: int32,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    dtime: float,
    cpd: float,
    iau_wgt_dyn: float,
    is_iau_active: bool,
    lhdiff_rcf: bool,
    divdamp_fac: float,
    divdamp_fac_o2: float,
    divdamp_order: int32,
    scal_divdamp_o2: float,
    limited_area: bool,
    itime_scheme: int32,
    istep: int32,
    horizontal_lower: int32,
    horizontal_upper: int32,
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
    nlev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    if istep == 1:
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
            idiv_method=idiv_method,
            igradp_method=igradp_method,
            horizontal_lower=horizontal_lower,
            horizontal_upper=horizontal_upper,
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
            nlev=nlev,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        )
    else:
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
            wgt_nnow_vel=wgt_nnow_vel,
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
            horizontal_lower=horizontal_lower,
            horizontal_upper=horizontal_upper,
            horizontal_lower_2=horizontal_lower_2,
            horizontal_upper_2=horizontal_upper_2,
            kstart_dd3d=kstart_dd3d,
            nlev=nlev,
        )
    return z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn


@program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_15_to_28(
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    p_vn: Field[[EdgeDim, KDim], float],
    p_vt: Field[[EdgeDim, KDim], float],
    pos_on_tplane_e_1: Field[[ECDim], float],
    pos_on_tplane_e_2: Field[[ECDim], float],
    primal_normal_cell_1: Field[[ECDim], float],
    dual_normal_cell_1: Field[[ECDim], float],
    primal_normal_cell_2: Field[[ECDim], float],
    dual_normal_cell_2: Field[[ECDim], float],
    rho_ref_me: Field[[EdgeDim, KDim], float],
    theta_ref_me: Field[[EdgeDim, KDim], float],
    z_rth_pr_1: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], float],
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vn_incr: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    horz_idx: Field[[EdgeDim], int32],
    vert_idx: Field[[KDim], int32],
    grav_o_cpd: float,
    p_dthalf: float,
    idiv_method: int32,
    igradp_method: int32,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    dtime: float,
    cpd: float,
    iau_wgt_dyn: float,
    is_iau_active: bool,
    lhdiff_rcf: bool,
    divdamp_fac: float,
    divdamp_fac_o2: float,
    divdamp_order: int32,
    scal_divdamp_o2: float,
    limited_area: bool,
    itime_scheme: int32,
    istep: int32,
    horizontal_lower: int32,
    horizontal_upper: int32,
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
    nlev: int32,
    nflatlev: int32,
    nflat_gradp: int32,
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
        vn_nnow=vn_nnow,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_phy=ddt_vn_phy,
        z_graddiv_vn=z_graddiv_vn,
        vn_incr=vn_incr,
        vn=vn,
        z_rho_e=z_rho_e,
        z_theta_v_e=z_theta_v_e,
        z_gradh_exner=z_gradh_exner,
        z_graddiv2_vn=z_graddiv2_vn,
        z_hydro_corr=z_hydro_corr,
        geofac_grdiv=geofac_grdiv,
        scal_divdamp=scal_divdamp,
        bdy_divdamp=bdy_divdamp,
        nudgecoeff_e=nudgecoeff_e,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        grav_o_cpd=grav_o_cpd,
        p_dthalf=p_dthalf,
        idiv_method=idiv_method,
        igradp_method=igradp_method,
        wgt_nnow_vel=wgt_nnow_vel,
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
        istep=istep,
        horizontal_lower=horizontal_lower,
        horizontal_upper=horizontal_upper,
        horizontal_lower_00=horizontal_lower_00,
        horizontal_upper_00=horizontal_upper_00,
        horizontal_lower_01=horizontal_lower_01,
        horizontal_upper_01=horizontal_upper_01,
        horizontal_lower_1=horizontal_lower_1,
        horizontal_upper_1=horizontal_upper_1,
        horizontal_lower_2=horizontal_lower_2,
        horizontal_upper_2=horizontal_upper_2,
        horizontal_lower_3=horizontal_lower_3,
        horizontal_upper_3=horizontal_upper_3,
        horizontal_lower_4=horizontal_lower_4,
        horizontal_upper_4=horizontal_upper_4,
        kstart_dd3d=kstart_dd3d,
        nlev=nlev,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        out=(z_rho_e, z_theta_v_e, z_gradh_exner, vn, z_graddiv_vn),
    )
