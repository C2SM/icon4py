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

from gt4py.next.ffront.decorator import GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where
from model.common.tests import field_aliases as fa

from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    wgtfacq_c: Field[[CellDim, KDim], vpfloat],
    vert_idx: fa.KintField,
    nlev: int32,
    nflatlev: int32,
) -> Field[[CellDim, KDim], vpfloat]:
    w_concorr_c = where(
        nflatlev + 1 <= vert_idx < nlev,
        _compute_contravariant_correction_of_w(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, z_w_concorr_me, wgtfacq_c
        ),
    )
    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    wgtfacq_c: Field[[CellDim, KDim], vpfloat],
    vert_idx: fa.KintField,
    nlev: int32,
    nflatlev: int32,
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_solve_nonhydro_stencil_39_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        wgtfacq_c,
        vert_idx,
        nlev,
        nflatlev,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
