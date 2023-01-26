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
from functional.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import C2E, C2EDim, CellDim, EdgeDim, KDim, Koff


@field_operator
def _neighbor_sum_into_koff_bug(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
) -> Field[[CellDim, KDim], float]:
    kh_c = neighbor_sum(kh_smag_ec(C2E) * e_bln_c_s, axis=C2EDim)
    hdef_ic = kh_c(Koff[-1])
    return hdef_ic


@program
def neighbor_sum_into_koff_bug(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
):
    _neighbor_sum_into_koff_bug(
        kh_smag_ec,
        e_bln_c_s,
        out=(hdef_ic),
    )
