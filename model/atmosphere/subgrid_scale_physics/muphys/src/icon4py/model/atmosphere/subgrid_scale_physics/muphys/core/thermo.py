# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.physics.thermodynamics import (
    T_from_internal_energy,
    T_from_internal_energy_scalar,
    _internal_energy,
    _internal_energy_scalar,
    _T_from_internal_energy,
    _T_from_internal_energy_scalar,
    internal_energy,
)
from icon4py.model.common.type_alias import wpfloat


__all__ = [
    # Re-exports of the AES thermodynamics helpers that moved to icon4py.model.common
    "T_from_internal_energy",
    "T_from_internal_energy_scalar",
    "_T_from_internal_energy",
    "_T_from_internal_energy_scalar",
    "_internal_energy",
    "_internal_energy_scalar",
    "internal_energy",
]


@gtx.field_operator
def _qsat_ice_rho(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the Qsat pressure due to ice

    Args:
        t:                  Temperature
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)
    C3IES = wpfloat(21.875)
    C4IES = wpfloat(7.66)

    return (C1ES * exp(C3IES * (t - ThermodynamicConsts.tmelt) / (t - C4IES))) / (
        rho * ThermodynamicConsts.rv * t
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_ice_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_ice_rho(t=t, rho=rho, out=pressure)


@gtx.field_operator
def _qsat_rho(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the standard Qsat pressure

    Args:
        t:                  Temperature
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)

    return (C1ES * exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))) / (
        rho * ThermodynamicConsts.rv * t
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho(t=t, rho=rho, out=pressure)


@gtx.field_operator
def _qsat_rho_tmelt(
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the Qsat pressure at t.melt

    Args:
        rho:                Density

    Result:                 Pressure
    """
    C1ES = wpfloat(610.78)

    return C1ES / (rho * ThermodynamicConsts.rv * ThermodynamicConsts.tmelt)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho_tmelt(
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho_tmelt(rho=rho, out=pressure)


@gtx.field_operator
def _dqsatdT_rho(
    qs: fa.CellKField[ta.wpfloat],
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the derivative of saturation vapor pressure (over liquid)

    Args:
        qs:                 Saturation vapor pressure (over liquid)
        t:                  Temperature

    Result:                 Derivative d(qsat_rho)/dT
    """
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)
    C5LES = C3LES * (ThermodynamicConsts.tmelt - C4LES)

    return qs * (C5LES / ((t - C4LES) * (t - C4LES)) - wpfloat(1.0) / t)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dqsatdT_rho(
    qs: fa.CellKField[ta.wpfloat],  # Saturation vapor pressure (over liquid)
    t: fa.CellKField[ta.wpfloat],  # Temperature
    derivative: fa.CellKField[ta.wpfloat],  # output
):
    _dqsatdT_rho(qs=qs, t=t, out=derivative)


@gtx.field_operator
def _sat_pres_ice(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the saturation pressure over ice

    Args:
        t:                    Temperature

    Result:                   Saturation pressure
    """
    C1ES = wpfloat(610.78)
    C3IES = wpfloat(21.875)
    C4IES = wpfloat(7.66)

    return C1ES * exp(C3IES * (t - ThermodynamicConsts.tmelt) / (t - C4IES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_ice(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _sat_pres_ice(t=t, out=pressure)


@gtx.field_operator
def _sat_pres_water(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the saturation pressure over water

    Args:
        t:                    Temperature

    Result:                   Saturation pressure
    """
    C1ES = wpfloat(610.78)
    C3LES = wpfloat(17.269)
    C4LES = wpfloat(35.86)

    return C1ES * exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_water(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
) -> None:
    _sat_pres_water(t=t, out=pressure)


@gtx.field_operator
def _newton_raphson(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
    Tx: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qve: fa.CellKField[ta.wpfloat],
    qce: fa.CellKField[ta.wpfloat],
    cvc: fa.CellKField[ta.wpfloat],
    ue: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute a Newton-Raphson iteration on the temperature

    Args:
        Tx:                    Temperature
        rho:                   Density containing dry air and water constituents
        qve:                   Specific humidity
        qre:                   Specific rain water
        cvc:                   Constant volume precalculation
        ue:                    Energy

    Result:                    Revised temperature
    """
    qx = _qsat_rho(Tx, rho)
    dqx = _dqsatdT_rho(qx, Tx)
    qcx = qve + qce - qx
    cv = cvc + ThermodynamicConsts.cvv * qx + ThermodynamicConsts.clw * qcx
    ux = cv * Tx - qcx * GraupelConsts.lvc
    dux = cv + dqx * (GraupelConsts.lvc + (ThermodynamicConsts.cvv - ThermodynamicConsts.clw) * Tx)
    Tx = Tx - (ux - ue) / dux
    return Tx
