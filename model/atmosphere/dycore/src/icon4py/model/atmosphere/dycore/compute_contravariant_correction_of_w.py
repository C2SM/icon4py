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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_contravariant_correction_of_w(
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_39."""
    z_w_concorr_me_offset_1 = z_w_concorr_me(Koff[-1])

    z_w_concorr_me_wp, z_w_concorr_me_offset_1_wp = astype(
        (z_w_concorr_me, z_w_concorr_me_offset_1), wpfloat
    )

    z_w_concorr_mc_m1_wp = neighbor_sum(
        e_bln_c_s(C2CE) * z_w_concorr_me_offset_1_wp(C2E), axis=C2EDim
    )
    z_w_concorr_mc_m0_wp = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_wp(C2E), axis=C2EDim)

    z_w_concorr_mc_m1_vp, z_w_concorr_mc_m0_vp = astype(
        (z_w_concorr_mc_m1_wp, z_w_concorr_mc_m0_wp), vpfloat
    )
    w_concorr_c_vp = (
        wgtfac_c * z_w_concorr_mc_m0_vp + (vpfloat("1.0") - wgtfac_c) * z_w_concorr_mc_m1_vp
    )
    return w_concorr_c_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_contravariant_correction_of_w(
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    wgtfac_c: Field[[CellDim, KDim], vpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_contravariant_correction_of_w(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
