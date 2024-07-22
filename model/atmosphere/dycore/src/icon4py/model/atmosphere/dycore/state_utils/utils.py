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

import gt4py.next as gtx

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _scale_k(field: gtx.Field[[KDim], float], factor: float) -> gtx.Field[[KDim], float]:
    return field * factor


@gtx.program(backend=backend)
def scale_k(field: gtx.Field[[KDim], float], factor: float, scaled_field: gtx.Field[[KDim], float]):
    _scale_k(field, factor, out=scaled_field)


@gtx.field_operator
def _broadcast_zero_to_three_edge_kdim_fields_wp() -> (
    tuple[
        gtx.Field[[EdgeDim, KDim], wpfloat],
        gtx.Field[[EdgeDim, KDim], wpfloat],
        gtx.Field[[EdgeDim, KDim], wpfloat],
    ]
):
    return (
        gtx.broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        gtx.broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        gtx.broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
    )


@gtx.field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: gtx.Field[[KDim], float], nudge_max_coeff: float, dbl_eps: float
) -> gtx.Field[[KDim], float]:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


@gtx.field_operator
def _calculate_scal_divdamp(
    enh_divdamp_fac: gtx.Field[[KDim], float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
) -> gtx.Field[[KDim], float]:
    enh_divdamp_fac = (
        gtx.maximum(0.0, enh_divdamp_fac - 0.25 * divdamp_fac_o2)
        if divdamp_order == 24
        else enh_divdamp_fac
    )
    return -enh_divdamp_fac * mean_cell_area**2


@gtx.field_operator
def _calculate_divdamp_fields(
    enh_divdamp_fac: gtx.Field[[KDim], float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
    nudge_max_coeff: float,
    dbl_eps: float,
) -> tuple[gtx.Field[[KDim], float], gtx.Field[[KDim], float]]:
    scal_divdamp = _calculate_scal_divdamp(
        enh_divdamp_fac, divdamp_order, mean_cell_area, divdamp_fac_o2
    )
    bdy_divdamp = _calculate_bdy_divdamp(scal_divdamp, nudge_max_coeff, dbl_eps)
    return (scal_divdamp, bdy_divdamp)


@gtx.field_operator
def _compute_z_raylfac(
    rayleigh_w: gtx.Field[[KDim], float], dtime: float
) -> gtx.Field[[KDim], float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@gtx.program(backend=backend)
def compute_z_raylfac(
    rayleigh_w: gtx.Field[[KDim], float], dtime: float, z_raylfac: gtx.Field[[KDim], float]
):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)
