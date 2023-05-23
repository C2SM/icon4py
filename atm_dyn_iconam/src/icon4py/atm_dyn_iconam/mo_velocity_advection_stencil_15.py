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
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next.program_processors.runners import gtfn_cpu

from icon4py.common.dimension import CellDim, KDim, Koff


@field_operator
def _mo_velocity_advection_stencil_15(
    z_w_con_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_w_con_c_full = 0.5 * (z_w_con_c + z_w_con_c(Koff[1]))
    return z_w_con_c_full


@program(backend=gtfn_cpu.run_gtfn)
def mo_velocity_advection_stencil_15(
    z_w_con_c: Field[[CellDim, KDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _mo_velocity_advection_stencil_15(
        z_w_con_c,
        out=z_w_con_c_full,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
