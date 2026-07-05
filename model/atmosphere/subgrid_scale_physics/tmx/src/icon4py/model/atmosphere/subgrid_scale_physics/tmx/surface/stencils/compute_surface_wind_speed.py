# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_wind_speed(
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
    reference_u: fa.CellField[wpfloat],
    reference_v: fa.CellField[wpfloat],
    min_sfc_wind: wpfloat,
) -> fa.CellField[wpfloat]:
    """
    Compute the surface-relative wind speed.

    Port of 'compute_wind_speed' (mo_vdf_diag_smag.f90:180-186):
    ``wind_rel = max(min_sfc_wind, |u_atm - u_ref|)``. The reference velocity is
    the ocean current over ocean, the ice drift over ice and zero over land.

    Args:
        ua: zonal wind at the lowest full level [m/s]
        va: meridional wind at the lowest full level [m/s]
        reference_u: zonal reference (surface) velocity [m/s]
        reference_v: meridional reference (surface) velocity [m/s]
        min_sfc_wind: minimum surface wind speed in the free-convection limit [m/s]

    Returns:
        surface-relative wind speed [m/s]
    """
    return maximum(min_sfc_wind, sqrt((ua - reference_u) ** 2 + (va - reference_v) ** 2))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_wind_speed(
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
    reference_u: fa.CellField[wpfloat],
    reference_v: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    min_sfc_wind: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_wind_speed(
        ua=ua,
        va=va,
        reference_u=reference_u,
        reference_v=reference_v,
        min_sfc_wind=min_sfc_wind,
        out=wind_rel,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
