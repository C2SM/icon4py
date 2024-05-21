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
def _compute_wgtfac_c_nlev(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_0(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[+1]) - z_ifc) / (z_ifc(Koff[+2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_inner(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-1]) - z_ifc(Koff[+1]))
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
) -> Field[[CellDim, KDim], wpfloat]:
    wgt_fac_c = where((k > 0) & (k < nlev), _compute_wgtfac_c_inner(z_ifc), z_ifc)
    wgt_fac_c = where(k == 0, _compute_wgtfac_c_0(z_ifc=z_ifc), wgt_fac_c)
    wgt_fac_c = where(k == nlev, _compute_wgtfac_c_nlev(z_ifc=z_ifc), wgt_fac_c)

    return wgt_fac_c


@program(grid_type=GridType.UNSTRUCTURED, backend=run_gtfn)
def compute_wgtfac_c(
    wgtfac_c: Field[[CellDim, KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k: Field[[KDim], int32],
    nlev: int32,
):
    _compute_wgtfac_c(
        z_ifc,
        k,
        nlev,
        out=wgtfac_c,
    )
