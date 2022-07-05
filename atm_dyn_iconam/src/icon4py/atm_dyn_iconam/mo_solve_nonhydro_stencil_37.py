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
from functional.ffront.fbuiltins import Field, float

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_37_vn_ie(
    vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    vn_ie = vn
    return vn_ie


@program
def mo_solve_nonhydro_stencil_37_vn_ie(
    vn: Field[[EdgeDim, KDim], float],
    out: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_37_vn_ie(vn, out=out)


@field_operator
def _mo_solve_nonhydro_stencil_37_z_vt_ie(
    vt: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_vt_ie = vt
    return z_vt_ie


@program
def mo_solve_nonhydro_stencil_37_z_vt_ie(
    vt: Field[[EdgeDim, KDim], float],
    out: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_37_z_vt_ie(vt, out=out)


@field_operator
def _mo_solve_nonhydro_stencil_37_z_kin_hor_e(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_kin_hor_e = 0.5 * (vn * vn + vt * vt)
    return z_kin_hor_e


@program
def mo_solve_nonhydro_stencil_37_z_kin_hor_e(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    out: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_37_z_kin_hor_e(vn, vt, out=out)


@program
def mo_solve_nonhydro_stencil_37(
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_37_vn_ie(vn, out=vn_ie)
    _mo_solve_nonhydro_stencil_37_z_vt_ie(vt, out=z_vt_ie)
    _mo_solve_nonhydro_stencil_37_z_kin_hor_e(vn, vt, out=z_kin_hor_e)
