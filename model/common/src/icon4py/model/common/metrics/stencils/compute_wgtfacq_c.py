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
from gt4py.next.ffront.fbuiltins import Field, int32, where
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_wgtfacq_c(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
) -> Field[[CellDim, KDim], wpfloat]:

    z1 = 0.5 * z_ifc(Koff[-1]) - z_ifc
    z2 = 0.5 * z_ifc(Koff[-1]) + z_ifc(Koff[-2]) - z_ifc
    z3 = 0.5 * z_ifc(Koff[-2]) + z_ifc(Koff[-3]) - z_ifc

    wgt_facq_c = where(k == int32(2), z1*z2/(z2-z3)/(z1-z3), wpfloat(0.0))
    wgt_facq_c = where(k == int32(1),z1-wgt_facq_c(Koff[1]) * (z1-z3)/(z1-z2), wgt_facq_c)
    wgt_facq_c = where(k == int32(0), 1.0 - wgt_facq_c(Koff[1]) + wgt_facq_c(Koff[2]), wgt_facq_c)


    return wgt_facq_c


@program(grid_type=GridType.UNSTRUCTURED, backend=run_gtfn)
def compute_wgtfacq_c(
    wgtfacq_c: Field[[CellDim, KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
):
    _compute_wgtfacq_c(
        z_ifc,
        k,
        out=wgtfacq_c,
    )

