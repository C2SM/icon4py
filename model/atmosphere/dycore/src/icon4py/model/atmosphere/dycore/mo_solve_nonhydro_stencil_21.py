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
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import E2C, E2EC, CellDim, ECDim, EdgeDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_21(
    theta_v: Field[[CellDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    grav_o_cpd: float,
) -> Field[[EdgeDim, KDim], float]:
    theta_v_0 = theta_v(as_offset(Koff, ikoffset(E2EC[0])))
    theta_v_1 = theta_v(as_offset(Koff, ikoffset(E2EC[1])))

    theta_v_ic_0 = theta_v_ic(as_offset(Koff, ikoffset(E2EC[0])))
    theta_v_ic_1 = theta_v_ic(as_offset(Koff, ikoffset(E2EC[1])))

    theta_v_ic_p1_0 = theta_v_ic(as_offset(Koff, ikoffset(E2EC[0]) + int32(1)))
    theta_v_ic_p1_1 = theta_v_ic(as_offset(Koff, ikoffset(E2EC[1]) + int32(1)))

    inv_ddqz_z_full_0 = inv_ddqz_z_full(as_offset(Koff, ikoffset(E2EC[0])))
    inv_ddqz_z_full_1 = inv_ddqz_z_full(as_offset(Koff, ikoffset(E2EC[1])))

    z_theta_0 = theta_v_0(E2C[0]) + zdiff_gradp(E2EC[0]) * (
        theta_v_ic_0(E2C[0]) - theta_v_ic_p1_0(E2C[0])
    ) * inv_ddqz_z_full_0(E2C[0])
    z_theta_1 = theta_v_1(E2C[1]) + zdiff_gradp(E2EC[1]) * (
        theta_v_ic_1(E2C[1]) - theta_v_ic_p1_1(E2C[1])
    ) * inv_ddqz_z_full_1(E2C[1])
    # TODO(magdalena): change exponent back to int (workaround for gt4py)
    z_hydro_corr = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta_1 - z_theta_0)
        * float(4.0)
        / ((z_theta_0 + z_theta_1) *(z_theta_0 + z_theta_1))
    )

    return z_hydro_corr


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_21(
    theta_v: Field[[CellDim, KDim], float],
    ikoffset: Field[[ECDim, KDim], int32],
    zdiff_gradp: Field[[ECDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    grav_o_cpd: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_21(
        theta_v,
        ikoffset,
        zdiff_gradp,
        theta_v_ic,
        inv_ddqz_z_full,
        inv_dual_edge_length,
        grav_o_cpd,
        out=z_hydro_corr,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
