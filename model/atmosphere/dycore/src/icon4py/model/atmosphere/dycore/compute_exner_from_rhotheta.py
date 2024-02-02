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
from gt4py.next.ffront.fbuiltins import Field, exp, int32, log

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_exner_from_rhotheta(
    rho: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    '''Formerly known as _mo_solve_nonhydro_stencil_67.'''
    theta_v_wp = exner
    exner_wp = exp(rd_o_cvd * log(rd_o_p0ref * rho * theta_v_wp))
    return theta_v_wp, exner_wp


@program(grid_type=GridType.UNSTRUCTURED)
def compute_exner_from_rhotheta(
    rho: Field[[CellDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    exner: Field[[CellDim, KDim], wpfloat],
    rd_o_cvd: wpfloat,
    rd_o_p0ref: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_exner_from_rhotheta(
        rho,
        theta_v,
        exner,
        rd_o_cvd,
        rd_o_p0ref,
        out=(theta_v, exner),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
