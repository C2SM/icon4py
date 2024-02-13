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

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


#          p_nh(jg)%metrics%wgtfac_c(1:nlen,nlevp1,jb) = &
#           (p_nh(jg)%metrics%z_ifc(1:nlen,nlev,jb) -    &
#            p_nh(jg)%metrics%z_ifc(1:nlen,nlevp1,jb)) / &
#           (p_nh(jg)%metrics%z_ifc(1:nlen,nlev-1,jb) -  &
#            p_nh(jg)%metrics%z_ifc(1:nlen,nlevp1,jb))
@field_operator
def _calc_wgtfac_c_nlevp1(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:

    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-2]) - z_ifc)
    return z_wgtfac_c


#          p_nh(jg)%metrics%wgtfac_c(1:nlen,1,jb) = &
#           (p_nh(jg)%metrics%z_ifc(1:nlen,2,jb) -  &
#            p_nh(jg)%metrics%z_ifc(1:nlen,1,jb)) / &
#           (p_nh(jg)%metrics%z_ifc(1:nlen,3,jb) -  &
#            p_nh(jg)%metrics%z_ifc(1:nlen,1,jb))
@field_operator
def _calc_wgtfac_c_0(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:

    z_wgtfac_c = (z_ifc(Koff[+1]) - z_ifc) / (z_ifc(Koff[+2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _calc_wgtfac_c_inner(
    z_ifc: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:

    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-1]) - z_ifc(Koff[+1]))
    return z_wgtfac_c


@field_operator
def _calc_wgtfac_c(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k_field: Field[[KDim], int32],
    nlevp1: int32,
) -> Field[[CellDim, KDim], wpfloat]:

    wgt_fac_c = where(
        (k_field > int32(0)) & (k_field < nlevp1-1), _calc_wgtfac_c_inner(z_ifc), z_ifc
    )
    wgt_fac_c = where(k_field == int32(0), _calc_wgtfac_c_0(z_ifc=z_ifc), wgt_fac_c)
    wgt_fac_c = where(k_field == nlevp1, _calc_wgtfac_c_nlevp1(z_ifc=z_ifc), wgt_fac_c)

    return wgt_fac_c


@program(grid_type=GridType.UNSTRUCTURED)
def calc_wgtfac_c(
    wgtfac_c: Field[[CellDim, KDim], wpfloat],
    z_ifc: Field[[CellDim, KDim], wpfloat],
    k_field: Field[[KDim], int32],
    nlevp1: int32,
):
    _calc_wgtfac_c(
        z_ifc,
        k_field,
        nlevp1,
        out=wgtfac_c,
    )
