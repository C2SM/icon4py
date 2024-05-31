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
from model.common.tests import field_type_aliases as fa

from icon4py.model.atmosphere.dycore.compute_avg_vn import _compute_avg_vn
from icon4py.model.atmosphere.dycore.compute_tangential_wind import _compute_tangential_wind
from icon4py.model.common.dimension import E2C2EO, E2C2EDim, E2C2EODim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_avg_vn_and_graddiv_vn_and_vt(
    e_flx_avg: Field[[EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EKwpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
) -> tuple[
    fa.EKwpField,
    fa.EKvpField,
    fa.EKvpField,
]:
    """Formerly known as _mo_solve_nonhydro_stencil_30."""
    z_vn_avg_wp = _compute_avg_vn(e_flx_avg=e_flx_avg, vn=vn)
    z_graddiv_vn_vp = astype(neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim), vpfloat)
    vt_vp = _compute_tangential_wind(vn=vn, rbf_vec_coeff_e=rbf_vec_coeff_e)
    return z_vn_avg_wp, z_graddiv_vn_vp, vt_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_avg_vn_and_graddiv_vn_and_vt(
    e_flx_avg: Field[[EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EKwpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
    z_vn_avg: fa.EKwpField,
    z_graddiv_vn: fa.EKvpField,
    vt: fa.EKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_avg_vn_and_graddiv_vn_and_vt(
        e_flx_avg,
        vn,
        geofac_grdiv,
        rbf_vec_coeff_e,
        out=(z_vn_avg, z_graddiv_vn, vt),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
