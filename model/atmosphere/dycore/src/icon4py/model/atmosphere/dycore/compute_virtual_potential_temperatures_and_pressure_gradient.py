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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.atmosphere.dycore.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: Field[[CellDim, KDim], vpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
) -> tuple[
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_09."""
    wgtfac_c_wp, ddqz_z_half_wp = astype((wgtfac_c, ddqz_z_half), wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_rth_pr_2)
    theta_v_ic_wp = wgtfac_c_wp * theta_v + (wpfloat("1.0") - wgtfac_c_wp) * theta_v(Koff[-1])
    z_th_ddz_exner_c_wp = vwind_expl_wgt * theta_v_ic_wp * (
        exner_pr(Koff[-1]) - exner_pr
    ) / ddqz_z_half_wp + astype(z_theta_v_pr_ic_vp * d_exner_dz_ref_ic, wpfloat)
    return z_theta_v_pr_ic_vp, theta_v_ic_wp, astype(z_th_ddz_exner_c_wp, vpfloat)


@field_operator
def _compute_only_virtual_potential_temperatures(
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: Field[[CellDim, KDim], vpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], vpfloat],
) -> tuple[
    Field[[CellDim, KDim], vpfloat],
    Field[[CellDim, KDim], wpfloat],
]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_rth_pr_2)
    theta_v_ic_wp = wgtfac_c_wp * theta_v + (wpfloat("1.0") - wgtfac_c_wp) * theta_v(Koff[-1])
    return z_theta_v_pr_ic_vp, theta_v_ic_wp


@field_operator
def _compute_only_pressure_gradient(
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    z_theta_v_pr_ic: Field[[CellDim, KDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)
    z_th_ddz_exner_c_wp = vwind_expl_wgt * theta_v_ic * (
        exner_pr(Koff[-1]) - exner_pr
    ) / ddqz_z_half_wp + astype(z_theta_v_pr_ic * d_exner_dz_ref_ic, wpfloat)
    return astype(z_th_ddz_exner_c_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    z_rth_pr_2: Field[[CellDim, KDim], vpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    exner_pr: Field[[CellDim, KDim], wpfloat],
    d_exner_dz_ref_ic: Field[[CellDim, KDim], vpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    z_theta_v_pr_ic: Field[[CellDim, KDim], vpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    z_th_ddz_exner_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_virtual_potential_temperatures_and_pressure_gradient(
        wgtfac_c,
        z_rth_pr_2,
        theta_v,
        vwind_expl_wgt,
        exner_pr,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=(z_theta_v_pr_ic, theta_v_ic, z_th_ddz_exner_c),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
