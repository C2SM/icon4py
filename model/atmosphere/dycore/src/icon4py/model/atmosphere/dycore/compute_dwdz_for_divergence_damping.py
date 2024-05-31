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
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_dwdz_for_divergence_damping(
    inv_ddqz_z_full: fa.CKvpField,
    w: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
) -> fa.CKvpField:
    """Formerly known as _mo_solve_nonhydro_stencil_56_63."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_dwdz_dd_wp = inv_ddqz_z_full_wp * (
        (w - w(Koff[1])) - astype(w_concorr_c - w_concorr_c(Koff[1]), wpfloat)
    )
    return astype(z_dwdz_dd_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_dwdz_for_divergence_damping(
    inv_ddqz_z_full: fa.CKvpField,
    w: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
    z_dwdz_dd: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_dwdz_for_divergence_damping(
        inv_ddqz_z_full,
        w,
        w_concorr_c,
        out=z_dwdz_dd,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
