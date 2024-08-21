# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _compute_results_for_thermodynamic_variables(
    z_rho_expl: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    rho_now: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    exner_now: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cvd_o_rd: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_55."""
    inv_ddqz_z_full_wp, exner_ref_mc_wp, z_alpha_wp, z_beta_wp = astype(
        (inv_ddqz_z_full, exner_ref_mc, z_alpha, z_beta), wpfloat
    )

    rho_new_wp = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full_wp * (
        rho_ic * w - rho_ic(Koff[1]) * w(Koff[1])
    )
    exner_new_wp = (
        z_exner_expl
        + exner_ref_mc_wp
        - z_beta_wp * (z_alpha_wp * w - z_alpha_wp(Koff[1]) * w(Koff[1]))
    )
    theta_v_new_wp = (
        rho_now
        * theta_v_now
        * ((exner_new_wp / exner_now - wpfloat("1.0")) * cvd_o_rd + wpfloat("1.0"))
        / rho_new_wp
    )
    return rho_new_wp, exner_new_wp, theta_v_new_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_results_for_thermodynamic_variables(
    z_rho_expl: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    rho_now: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    exner_now: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    theta_v_new: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cvd_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_results_for_thermodynamic_variables(
        z_rho_expl,
        vwind_impl_wgt,
        inv_ddqz_z_full,
        rho_ic,
        w,
        z_exner_expl,
        exner_ref_mc,
        z_alpha,
        z_beta,
        rho_now,
        theta_v_now,
        exner_now,
        dtime,
        cvd_o_rd,
        out=(rho_new, exner_new, theta_v_new),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
