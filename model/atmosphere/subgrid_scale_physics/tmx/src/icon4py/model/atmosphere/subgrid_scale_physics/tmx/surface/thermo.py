# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


"""
Surface thermodynamic helper field operators (2D cell fields).

Port of the elemental functions of ICON's mo_aes_thermo.f90 used by the tmx
surface scheme. The formulas and constants match the host functions in
icon4py.model.common.thermodynamic_functions and the (3D) muphys operators;
GT4Py field operators are dimension-monomorphic, so the surface uses these 2D
CellField specializations.
"""


@gtx.field_operator
def _sat_pres_water(t: fa.CellField[wpfloat]) -> fa.CellField[wpfloat]:
    """Saturation vapour pressure over liquid water [Pa] (Tetens; mo_aes_thermo.f90)."""
    return wpfloat(610.78) * exp(
        wpfloat(17.269) * (t - PhysicsConstants.tmelt) / (t - wpfloat(35.86))
    )


@gtx.field_operator
def _sat_pres_ice(t: fa.CellField[wpfloat]) -> fa.CellField[wpfloat]:
    """Saturation vapour pressure over ice [Pa] (Tetens; mo_aes_thermo.f90)."""
    return wpfloat(610.78) * exp(
        wpfloat(21.875) * (t - PhysicsConstants.tmelt) / (t - wpfloat(7.66))
    )


@gtx.field_operator
def _specific_humidity(
    vapor_pressure: fa.CellField[wpfloat], pressure: fa.CellField[wpfloat]
) -> fa.CellField[wpfloat]:
    """Specific humidity from vapour and total pressure [kg/kg] (mo_aes_thermo.f90)."""
    return (
        PhysicsConstants.rd_o_rv
        * vapor_pressure
        / (pressure - (wpfloat(1.0) - PhysicsConstants.rd_o_rv) * vapor_pressure)
    )


@gtx.field_operator
def _potential_temperature(
    t: fa.CellField[wpfloat], pressure: fa.CellField[wpfloat]
) -> fa.CellField[wpfloat]:
    """Dry potential temperature T*(p0ref/p)^(rd/cpd) [K] (mo_aes_thermo.f90)."""
    return t * (PhysicsConstants.p0ref / pressure) ** PhysicsConstants.rd_o_cpd
