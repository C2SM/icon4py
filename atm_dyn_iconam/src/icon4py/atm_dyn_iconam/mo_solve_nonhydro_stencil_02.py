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
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_02_z_exner_ex_pr(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_exner_ex_pr = (1.0 + exner_exfac) * (
        exner - exner_ref_mc
    ) - exner_exfac * exner_pr
    return z_exner_ex_pr


@field_operator
def _mo_solve_nonhydro_stencil_02_exner_pr(
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    exner_pr = exner - exner_ref_mc
    return exner_pr


@program
def mo_solve_nonhydro_stencil_02(
    exner_exfac: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    exner_ref_mc: Field[[CellDim, KDim], float],
    exner_pr: Field[[CellDim, KDim], float],
    z_exner_ex_pr: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_02_z_exner_ex_pr(
        exner_exfac, exner, exner_ref_mc, exner_pr, out=z_exner_ex_pr
    )
    _mo_solve_nonhydro_stencil_02_exner_pr(exner, exner_ref_mc, out=exner_pr)
