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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset

from icon4py.common.dimension import CellDim, KDim

Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _mo_solve_nonhydro_stencil_09(
    wgtfac_c: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_theta_v_pr_ic = wgtfac_c * z_rth_pr_2 + (1.0 - wgtfac_c) * z_rth_pr_2(Koff[-1])
    theta_v_ic = wgtfac_c * theta_v + (1.0 - wgtfac_c) * theta_v(Koff[-1])
    z_th_ddz_exner_c = (
        vwind_expl_wgt * theta_v_ic * (exner_pr(Koff[-1]) - exner_pr) / ddqz_z_half
        + z_theta_v_pr_ic * d_exner_dz_ref_ic
    )
    return z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c


@field_operator
def _mo_solve_nonhydro_stencil_09_z_theta_v_pr_ic(
    wgtfac_c: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_09(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
    )[0]


@field_operator
def _mo_solve_nonhydro_stencil_09_theta_v_ic(
    wgtfac_c: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_09(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
    )[1]


@field_operator
def _mo_solve_nonhydro_stencil_09_z_th_ddz_exner_c(
    wgtfac_c: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_09(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
    )[2]


@program
def mo_solve_nonhydro_stencil_09(
    wgtfac_c: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], float],
    ddqz_z_half: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    z_th_ddz_exner_c: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_09_z_theta_v_pr_ic(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=z_theta_v_pr_ic,
    )
    _mo_solve_nonhydro_stencil_09_theta_v_ic(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=theta_v_ic,
    )
    _mo_solve_nonhydro_stencil_09_z_th_ddz_exner_c(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=z_th_ddz_exner_c,
    )
