
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import exp, maximum, where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import g_ct, t_d
from icon4py.model.common import field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _T_from_internal_energy(
    u: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the temperature from the internal energy

    Args:
        u:                  Internal energy (extensive)
        qv:                 Water vapor specific humidity
        qliq:               Specific mass of liquid phases
        qice:               Specific mass of solid phases
        rho:                Ambient density
        dz:                 Extent of grid cell

    Return:                 Temperature
    """
    qtot = qliq + qice + qv  # total water specific mass
    cv = (
        (t_d.cvd * (1.0 - qtot) + t_d.cvv * qv + t_d.clw * qliq + g_ct.ci * qice) * rho * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * g_ct.lvc + qice * g_ct.lsc)) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy(
    u: fa.CellKField[ta.wpfloat],  # Internal energy (extensive)
    qv: fa.CellKField[ta.wpfloat],  # Water vapor specific humidity
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    temperature: fa.CellKField[ta.wpfloat],  # output
):
    _T_from_internal_energy(u, qv, qliq, qice, rho, dz, out=temperature)


@gtx.field_operator
def _T_from_internal_energy_scalar(
    u: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the temperature from the internal energy (scalar version callable from scan_operator)

    Args:
        u:                  Internal energy (extensive)
        qv:                 Water vapor specific humidity
        qliq:               Specific mass of liquid phases
        qice:               Specific mass of solid phases
        rho:                Ambient density
        dz:                 Extent of grid cell

    Return:                 Temperature
    """
    qtot = qliq + qice + qv  # total water specific mass
    cv = (
        (t_d.cvd * (1.0 - qtot) + t_d.cvv * qv + t_d.clw * qliq + g_ct.ci * qice) * rho * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * g_ct.lvc + qice * g_ct.lsc)) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy_scalar(
    u: ta.wpfloat,  # Internal energy (extensive)
    qv: ta.wpfloat,  # Water vapor specific humidity
    qliq: ta.wpfloat,  # Specific mass of liquid phases
    qice: ta.wpfloat,  # Specific mass of solid phases
    rho: ta.wpfloat,  # Ambient density
    dz: ta.wpfloat,  # Extent of grid cell
    temperature: ta.wpfloat,  # output
):
    _T_from_internal_energy_scalar(u, qv, qliq, qice, rho, dz, out=temperature)


@gtx.field_operator
def _internal_energy(
    t: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the internal energy from the temperature

    Args:
        t:                 Temperature
        qv:                Specific mass of vapor
        qliq:              Specific mass of liquid phases
        qice:              Specific mass of solid phases
        rho:               Ambient density
        dz:                Extent of grid cell

    Result:                Internal energy
    """
    qtot = qliq + qice + qv
    cv = t_d.cvd * (1.0 - qtot) + t_d.cvv * qv + t_d.clw * qliq + g_ct.ci * qice

    return rho * dz * (cv * t - qliq * g_ct.lvc - qice * g_ct.lsc)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def internal_energy(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qv: fa.CellKField[ta.wpfloat],  # Specific mass of vapor
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    energy: fa.CellKField[ta.wpfloat],  # output
):
    _internal_energy(t, qv, qliq, qice, rho, dz, out=energy)


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
    C1ES = 610.78
    C3IES = 21.875
    C4IES = 7.66

    return (C1ES * exp(C3IES * (t - t_d.tmelt) / (t - C4IES))) / (rho * t_d.rv * t)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_ice_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_ice_rho(t, rho, out=pressure)


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
    C1ES = 610.78
    C3LES = 17.269
    C4LES = 35.86

    return (C1ES * exp(C3LES * (t - t_d.tmelt) / (t - C4LES))) / (rho * t_d.rv * t)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho(t, rho, out=pressure)


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
    C1ES = 610.78

    return C1ES / (rho * t_d.rv * t_d.tmelt)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def qsat_rho_tmelt(
    rho: fa.CellKField[ta.wpfloat],  # Density
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _qsat_rho_tmelt(rho, out=pressure)


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
    C3LES = 17.269
    C4LES = 35.86
    C5LES = C3LES * (t_d.tmelt - C4LES)

    return qs * (C5LES / ((t - C4LES) * (t - C4LES)) - 1.0 / t)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dqsatdT_rho(
    qs: fa.CellKField[ta.wpfloat],  # Saturation vapor pressure (over liquid)
    t: fa.CellKField[ta.wpfloat],  # Temperature
    derivative: fa.CellKField[ta.wpfloat],  # output
):
    _dqsatdT_rho(qs, t, out=derivative)


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
    C1ES = 610.78
    C3IES = 21.875
    C4IES = 7.66

    return C1ES * exp(C3IES * (t - t_d.tmelt) / (t - C4IES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_ice(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _sat_pres_ice(t, out=pressure)


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
    C1ES = 610.78
    C3LES = 17.269
    C4LES = 35.86

    return C1ES * exp(C3LES * (t - t_d.tmelt) / (t - C4LES))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def sat_pres_water(
    t: fa.CellKField[ta.wpfloat],  # Temperature
    pressure: fa.CellKField[ta.wpfloat],  # output
):
    _sat_pres_water(t, out=pressure)


@gtx.field_operator
def _newton_raphson(
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
    cv = cvc + t_d.cvv * qx + t_d.clw * qcx
    ux = cv * Tx - qcx * g_ct.lvc
    dux = cv + dqx * (g_ct.lvc + (t_d.cvv - t_d.clw) * Tx)
    Tx = Tx - (ux - ue) / dux
    return Tx


@gtx.field_operator
def _saturation_adjustment(
    te: fa.CellKField[ta.wpfloat],
    qve: fa.CellKField[ta.wpfloat],
    qce: fa.CellKField[ta.wpfloat],
    qre: fa.CellKField[ta.wpfloat],
    qti: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[bool],
]:
    """
    Compute the saturation adjustment which revises internal energy and water contents

    Args:
        Tx:                    Temperature
        qve:                   Specific humidity
        qce:                   Specific cloud water content
        qre:                   Specific rain water
        qti:                   Specific mass of all ice species (total-ice)
        rho:                   Density containing dry air and water constituents

    Result:                    Tuple containing
                               - Revised temperature
                               - Revised specific cloud water content
                               - Revised specific vapor content
                               - Mask specifying where qce+qve less than holding capacity
    """
    qt = qve + qce + qre + qti
    cvc = t_d.cvd * (1.0 - qt) + t_d.clw * qre + g_ct.ci * qti
    cv = cvc + t_d.cvv * qve + t_d.clw * qce
    ue = cv * te - qce * g_ct.lvc
    Tx_hold = ue / (cv + qce * (t_d.cvv - t_d.clw))
    qx_hold = _qsat_rho(Tx_hold, rho)

    Tx = te
    # Newton-Raphson iteration: 6 times the same operations
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)

    # At this point we hope Tx has converged
    qx = _qsat_rho(Tx, rho)

    # Is it possible to unify the where for all three outputs??
    mask = qve + qce <= qx_hold
    te = where((qve + qce <= qx_hold), Tx_hold, Tx)
    qce = where((qve + qce <= qx_hold), 0.0, maximum(qve + qce - qx, 0.0))
    qve = where((qve + qce <= qx_hold), qve + qce, qx)

    return te, qve, qce, mask


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def saturation_adjustment(
    te: fa.CellKField[ta.wpfloat],  # Temperature
    qve: fa.CellKField[ta.wpfloat],  # Specific humidity
    qce: fa.CellKField[ta.wpfloat],  # Specific cloud water content
    qre: fa.CellKField[ta.wpfloat],  # Specific rain water
    qti: fa.CellKField[ta.wpfloat],  # Specific mass of all ice species (total-ice)
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    te_out: fa.CellKField[ta.wpfloat],  # Temperature
    qve_out: fa.CellKField[ta.wpfloat],  # Specific humidity
    qce_out: fa.CellKField[ta.wpfloat],  # Specific cloud water content
    mask_out: fa.CellKField[bool],  # Specific cloud water content
):
    _saturation_adjustment(te, qve, qce, qre, qti, rho, out=(te_out, qve_out, qce_out, mask_out))
