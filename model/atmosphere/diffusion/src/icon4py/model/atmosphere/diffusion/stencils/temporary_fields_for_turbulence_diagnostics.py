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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    geofac_div: Field[[CEDim], float],
    diff_multfac_smag: Field[[KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    kh_c = neighbor_sum(kh_smag_ec(C2E) * e_bln_c_s(C2CE), axis=C2EDim) / diff_multfac_smag
    div = neighbor_sum(vn(C2E) * geofac_div(C2CE), axis=C2EDim)
    return kh_c, div


@program(grid_type=GridType.UNSTRUCTURED)
def temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    geofac_div: Field[[CEDim], float],
    diff_multfac_smag: Field[[KDim], float],
    kh_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
):
    _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag, out=(kh_c, div)
    )
