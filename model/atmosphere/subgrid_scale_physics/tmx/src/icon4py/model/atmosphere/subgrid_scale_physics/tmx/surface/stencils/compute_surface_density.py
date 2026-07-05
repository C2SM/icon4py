# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_density(
    surface_pressure: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    Compute the surface air density.

    Port of 'compute_sfc_density' (mo_vdf_diag_smag.f90:116):
    ``rho = psfc / (rd * T_sfc * (1 + vtmpc1 * qsat_sfc))`` with the moist
    correction ``vtmpc1 = rv/rd - 1``.

    Args:
        surface_pressure: surface pressure [Pa]
        temperature_sfc: surface temperature [K]
        qsat_sfc: surface saturation specific humidity [kg/kg]

    Returns:
        surface air density [kg/m^3]
    """
    return surface_pressure / (
        PhysicsConstants.rd
        * temperature_sfc
        * (wpfloat(1.0) + PhysicsConstants.rv_o_rd_minus_1 * qsat_sfc)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_density(
    surface_pressure: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    rho_sfc: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_density(
        surface_pressure=surface_pressure,
        temperature_sfc=temperature_sfc,
        qsat_sfc=qsat_sfc,
        out=rho_sfc,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
