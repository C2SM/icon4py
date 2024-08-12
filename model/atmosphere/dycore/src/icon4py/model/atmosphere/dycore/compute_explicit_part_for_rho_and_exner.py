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
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _compute_explicit_part_for_rho_and_exner(
    rho_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_48 or _mo_solve_nonhydro_stencil_49."""
    inv_ddqz_z_full_wp, z_flxdiv_mass_wp, z_beta_wp, z_flxdiv_theta_wp, ddt_exner_phy_wp = astype(
        (inv_ddqz_z_full, z_flxdiv_mass, z_beta, z_flxdiv_theta, ddt_exner_phy), wpfloat
    )

    z_rho_expl_wp = rho_nnow - dtime * inv_ddqz_z_full_wp * (
        z_flxdiv_mass_wp + z_contr_w_fl_l - z_contr_w_fl_l(Koff[1])
    )

    z_exner_expl_wp = (
        exner_pr
        - z_beta_wp
        * (
            z_flxdiv_theta_wp
            + theta_v_ic * z_contr_w_fl_l
            - theta_v_ic(Koff[1]) * z_contr_w_fl_l(Koff[1])
        )
        + dtime * ddt_exner_phy_wp
    )
    return z_rho_expl_wp, z_exner_expl_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_explicit_part_for_rho_and_exner(
    z_rho_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    rho_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    exner_pr: fa.CellKField[wpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
