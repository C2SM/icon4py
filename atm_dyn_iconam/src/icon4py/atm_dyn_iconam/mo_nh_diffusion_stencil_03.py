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
from functional.ffront.fbuiltins import Field, FieldOffset

from icon4py.common.dimension import CellDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _mo_nh_diffusion_stencil_03(
    wgtfac_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    div_ic = wgtfac_c * div + (1.0 - wgtfac_c) * div(Koff[-1])
    hdef_ic = (wgtfac_c * kh_c + (1.0 - wgtfac_c) * kh_c(Koff[-1])) ** 2
    return div_ic, hdef_ic


@field_operator
def _mo_nh_diffusion_stencil_03_hdef_ic(
    wgtfac_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_nh_diffusion_stencil_03(wgtfac_c, div, kh_c)[1]


@field_operator
def _mo_nh_diffusion_stencil_03_div_ic(
    wgtfac_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_nh_diffusion_stencil_03(wgtfac_c, div, kh_c)[0]


@program
def mo_nh_diffusion_stencil_03(
    wgtfac_c: Field[[CellDim, KDim], float],
    kh_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_03_div_ic(wgtfac_c, div, kh_c, out=div_ic[:, 1:])
    _mo_nh_diffusion_stencil_03_hdef_ic(wgtfac_c, div, kh_c, out=hdef_ic[:, 1:])
