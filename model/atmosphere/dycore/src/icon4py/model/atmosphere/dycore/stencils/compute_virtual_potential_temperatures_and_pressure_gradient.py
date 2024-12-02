# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_vp import (
    _interpolate_to_half_levels_vp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_half_levels_wp import (
    _interpolate_to_half_levels_wp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_09."""
    wgtfac_c_wp, ddqz_z_half_wp = astype((wgtfac_c, ddqz_z_half), wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_rth_pr_2)
    theta_v_ic_wp = _interpolate_to_half_levels_wp(wgtfac_c=wgtfac_c_wp, interpolant=theta_v)
    z_th_ddz_exner_c_wp = vwind_expl_wgt * theta_v_ic_wp * (
        exner_pr(Koff[-1]) - exner_pr
    ) / ddqz_z_half_wp + astype(z_theta_v_pr_ic_vp * d_exner_dz_ref_ic, wpfloat)
    return z_theta_v_pr_ic_vp, theta_v_ic_wp, astype(z_th_ddz_exner_c_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _compute_virtual_potential_temperatures(
    wgtfac_c: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=z_rth_pr_2)
    theta_v_ic_wp = wgtfac_c_wp * theta_v + (wpfloat("1.0") - wgtfac_c_wp) * theta_v(Koff[-1])
    return z_theta_v_pr_ic_vp, theta_v_ic_wp


@field_operator
def _compute_pressure_gradient(
    vwind_expl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    z_theta_v_pr_ic: fa.CellKField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)
    z_th_ddz_exner_c_wp = vwind_expl_wgt * theta_v_ic * (
        exner_pr(Koff[-1]) - exner_pr
    ) / ddqz_z_half_wp + astype(z_theta_v_pr_ic * d_exner_dz_ref_ic, wpfloat)
    return astype(z_th_ddz_exner_c_wp, vpfloat)
