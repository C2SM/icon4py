# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_stress_land(
    rho_sfc: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute the land surface momentum stress.

    Port of the land branch of 'compute_sfc_fluxes' (mo_tmx_surface.f90:857-858).
    Over land the heat/moisture fluxes are prescribed (JSBACH cut line); only the
    momentum stress is bulk-computed, relative to a resting surface:

        ustress = rho * km * wind_rel * ua
        vstress = rho * km * wind_rel * va

    Args:
        rho_sfc: surface air density [kg/m^3]
        km: momentum transfer coefficient [-]
        wind_rel: surface-relative wind speed [m/s]
        ua: zonal wind at the lowest full level [m/s]
        va: meridional wind at the lowest full level [m/s]

    Returns:
        (u_stress, v_stress) [N/m^2]
    """
    u_stress = rho_sfc * km * wind_rel * ua
    v_stress = rho_sfc * km * wind_rel * va
    return u_stress, v_stress


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_stress_land(
    rho_sfc: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    ua: fa.CellField[wpfloat],
    va: fa.CellField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_stress_land(
        rho_sfc=rho_sfc,
        km=km,
        wind_rel=wind_rel,
        ua=ua,
        va=va,
        out=(u_stress, v_stress),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
