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

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _extrapolate_temporally_exner_pressure(
    exner_exfac: fa.CKvpField,
    exner: fa.CKwpField,
    exner_ref_mc: fa.CKvpField,
    exner_pr: fa.CKwpField,
) -> tuple[fa.CKvpField, fa.CKwpField]:
    """Formerly known as _mo_solve_nonhydro_stencil_02."""
    exner_exfac_wp, exner_ref_mc_wp = astype((exner_exfac, exner_ref_mc), wpfloat)

    z_exner_ex_pr_wp = (wpfloat("1.0") + exner_exfac_wp) * (
        exner - exner_ref_mc_wp
    ) - exner_exfac_wp * exner_pr
    exner_pr_wp = exner - exner_ref_mc_wp
    return astype(z_exner_ex_pr_wp, vpfloat), exner_pr_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def extrapolate_temporally_exner_pressure(
    exner_exfac: fa.CKvpField,
    exner: fa.CKwpField,
    exner_ref_mc: fa.CKvpField,
    exner_pr: fa.CKwpField,
    z_exner_ex_pr: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _extrapolate_temporally_exner_pressure(
        exner_exfac,
        exner,
        exner_ref_mc,
        exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
