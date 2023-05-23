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
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.common.dimension import (
    E2C,
    E2EC,
    E2V,
    CellDim,
    E2CDim,
    E2VDim,
    ECDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
)


@field_operator
def _mo_velocity_advection_stencil_19(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    zeta: Field[[VertexDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    ddt_vn_adv = -(
        (coeff_gradekin(E2EC[0]) - coeff_gradekin(E2EC[1])) * z_kin_hor_e
        + (
            -coeff_gradekin(E2EC[0]) * z_ekinh(E2C[0])
            + coeff_gradekin(E2EC[1]) * z_ekinh(E2C[1])
        )
        + vt * (f_e + 0.5 * neighbor_sum(zeta(E2V), axis=E2VDim))
        + neighbor_sum(z_w_con_c_full(E2C) * c_lin_e, axis=E2CDim)
        * (vn_ie - vn_ie(Koff[1]))
        / ddqz_z_full_e
    )
    return ddt_vn_adv


@program(backend=gtfn_cpu.run_gtfn)
def mo_velocity_advection_stencil_19(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    zeta: Field[[VertexDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    ddt_vn_adv: Field[[EdgeDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _mo_velocity_advection_stencil_19(
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        out=ddt_vn_adv,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
