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
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: Field[[EdgeDim, KDim], vpfloat],
    vn: fa.EKwpField,
    e_bln_c_s: Field[[CEDim], wpfloat],
    geofac_div: Field[[CEDim], wpfloat],
    diff_multfac_smag: Field[[KDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], vpfloat]]:
    kh_smag_ec_wp, diff_multfac_smag_wp = astype((kh_smag_ec, diff_multfac_smag), wpfloat)

    kh_c_wp = neighbor_sum(kh_smag_ec_wp(C2E) * e_bln_c_s(C2CE), axis=C2EDim) / diff_multfac_smag_wp
    div_wp = neighbor_sum(vn(C2E) * geofac_div(C2CE), axis=C2EDim)
    return astype((kh_c_wp, div_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: Field[[EdgeDim, KDim], vpfloat],
    vn: fa.EKwpField,
    e_bln_c_s: Field[[CEDim], wpfloat],
    geofac_div: Field[[CEDim], wpfloat],
    diff_multfac_smag: Field[[KDim], vpfloat],
    kh_c: Field[[CellDim, KDim], vpfloat],
    div: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec,
        vn,
        e_bln_c_s,
        geofac_div,
        diff_multfac_smag,
        out=(kh_c, div),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
