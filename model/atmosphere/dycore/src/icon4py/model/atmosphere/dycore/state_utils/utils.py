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
from gt4py.next.ffront.fbuiltins import (
    abs,
    broadcast,
    maximum,
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _scale_k(field: fa.KField[float], factor: float) -> fa.KField[float]:
    return field * factor


@gtx.program(backend=backend)
def scale_k(field: fa.KField[float], factor: float, scaled_field: fa.KField[float]):
    _scale_k(field, factor, out=scaled_field)


@gtx.field_operator
def _broadcast_zero_to_three_edge_kdim_fields_wp() -> (
    tuple[
        fa.EdgeKField[wpfloat],
        fa.EdgeKField[wpfloat],
        fa.EdgeKField[wpfloat],
    ]
):
    return (
        broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
        broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
        broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
    )


@gtx.field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: fa.KField[float], nudge_max_coeff: float, dbl_eps: float
) -> fa.KField[float]:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


@gtx.field_operator
def _calculate_scal_divdamp(
    enh_divdamp_fac: fa.KField[float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
) -> fa.KField[float]:
    enh_divdamp_fac = (
        maximum(0.0, enh_divdamp_fac - 0.25 * divdamp_fac_o2)
        if divdamp_order == 24
        else enh_divdamp_fac
    )
    return -enh_divdamp_fac * mean_cell_area**2


@gtx.field_operator
def _calculate_divdamp_fields(
    enh_divdamp_fac: fa.KField[float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    divdamp_fac_o2: float,
    nudge_max_coeff: float,
    dbl_eps: float,
) -> tuple[fa.KField[float], fa.KField[float]]:
    scal_divdamp = _calculate_scal_divdamp(
        enh_divdamp_fac, divdamp_order, mean_cell_area, divdamp_fac_o2
    )
    bdy_divdamp = _calculate_bdy_divdamp(scal_divdamp, nudge_max_coeff, dbl_eps)
    return (scal_divdamp, bdy_divdamp)


@gtx.field_operator
def _compute_z_raylfac(rayleigh_w: fa.KField[float], dtime: float) -> fa.KField[float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@gtx.program(backend=backend)
def compute_z_raylfac(rayleigh_w: fa.KField[float], dtime: float, z_raylfac: fa.KField[float]):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)
