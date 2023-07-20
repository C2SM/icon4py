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
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_30 import (
    _mo_solve_nonhydro_stencil_30,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_31 import (
    _mo_solve_nonhydro_stencil_31,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_32 import (
    _mo_solve_nonhydro_stencil_32,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_34 import (
    _mo_solve_nonhydro_stencil_34,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_35 import (
    _mo_solve_nonhydro_stencil_35,
)
from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_36 import (
    _mo_solve_nonhydro_stencil_36,
)
from icon4py.common.dimension import E2C2EDim, E2C2EODim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_30_to_38(
    istep: int32,
    lclean_mflx: bool,
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    vn_traj: Field[[EdgeDim, KDim], float],
    mass_flx_me: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    r_nsubsteps: float,
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    z_vn_avg, z_graddiv_vn, vt = (
        _mo_solve_nonhydro_stencil_30(e_flx_avg, vn, geofac_grdiv, rbf_vec_coeff_e)
        if istep == 0
        else (
            _mo_solve_nonhydro_stencil_31(e_flx_avg, vn),
            broadcast(0.0, (EdgeDim, KDim)),
            broadcast(0.0, (EdgeDim, KDim)),
        )
    )  # if itime_scheme >= 5 not implemented
    mass_fl_e, z_theta_v_fl_e = _mo_solve_nonhydro_stencil_32(
        z_rho_e, z_vn_avg, ddqz_z_full_e, z_theta_v_e
    )

    vn_traj, mass_flx_me = (
        _mo_solve_nonhydro_stencil_34(
            z_vn_avg,
            mass_fl_e,
            broadcast(0.0, (EdgeDim, KDim)),
            broadcast(0.0, (EdgeDim, KDim)),
            r_nsubsteps,
        )
        if lclean_mflx
        else _mo_solve_nonhydro_stencil_34(
            z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps
        )
    )

    z_w_concorr_me = _mo_solve_nonhydro_stencil_35(vn, ddxn_z_full, ddxt_z_full, vt)
    vn_ie, z_vt_ie, z_kin_hor_e = _mo_solve_nonhydro_stencil_36(
        wgtfac_e,
        vn,
        vt,
    )

    return (
        z_vn_avg,
        z_graddiv_vn,
        vt,
        mass_fl_e,
        z_theta_v_fl_e,
        vn_traj,
        mass_flx_me,
        z_w_concorr_me,
        vn_ie,
        z_vt_ie,
        z_kin_hor_e,
    )


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_30_to_38(
    istep: int32,  # zero-based
    lclean_mflx: bool,
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    z_rho_e: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    ddxn_z_full: Field[[EdgeDim, KDim], float],
    ddxt_z_full: Field[[EdgeDim, KDim], float],
    wgtfac_e: Field[[EdgeDim, KDim], float],
    z_vn_avg: Field[[EdgeDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    vn_traj: Field[[EdgeDim, KDim], float],
    mass_flx_me: Field[[EdgeDim, KDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    r_nsubsteps: float,
):
    _mo_solve_nonhydro_stencil_30_to_38(
        istep,
        lclean_mflx,
        e_flx_avg,
        vn,
        geofac_grdiv,
        rbf_vec_coeff_e,
        z_rho_e,
        ddqz_z_full_e,
        z_theta_v_e,
        vn_traj,
        mass_flx_me,
        ddxn_z_full,
        ddxt_z_full,
        wgtfac_e,
        r_nsubsteps,
        out=(
            z_vn_avg,
            z_graddiv_vn,
            vt,
            mass_fl_e,
            z_theta_v_fl_e,
            vn_traj,
            mass_flx_me,
            z_w_concorr_me,
            vn_ie,
            z_vt_ie,
            z_kin_hor_e,
        ),
    )
