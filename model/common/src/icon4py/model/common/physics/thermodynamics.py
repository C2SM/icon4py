# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
AES thermodynamic helper functions shared across physics parameterizations.

This module implements the ``internal_energy`` and ``T_from_internal_energy``
functions of ICON's ``mo_aes_thermo.f90``. They were originally ported as part
of the muphys (graupel) microphysics and were promoted to ``icon4py.model.common``
so that other parameterizations (e.g. the AES turbulent mixing energy diffusion)
can use them without depending on muphys.

The constants are kept as members of a ``wpfloat``-based ``enum.Enum``
(a GT4Py/gtfn requirement for symbols referenced inside field operators) with
the exact literal values used by muphys, so results are bit-identical with the
original muphys implementation. Note that some of these values differ in the
last bit from the derived values in :mod:`icon4py.model.common.constants`
(e.g. ``cvd = 717.60`` vs. ``CPD - RD``), hence they are not shared.
"""

import enum

import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


class ThermodynamicConstants(ta.wpfloat, enum.Enum):
    """Thermodynamic constants of mo_aes_thermo.f90 (values as used by muphys)."""

    cvd = 717.60  # [J/K/kg] specific heat of dry air at constant volume => cpd - rd
    cvv = 1407.95  # [J/K/kg] specific heat of water vapor at constant volume => cpv - rv
    clw = 4192.6641119999995  # specific heat capacity of liquid water => (rcpl + 1.0) * cpd
    ci = 2108.0  # specific heat of ice
    lvc = 3135383.2031928  # invariant part of vaporization enthalpy => alv - (cpv - clw) * tmelt
    lsc = 2899657.201  # invariant part of sublimation enthalpy => als - (cpv - ci) * tmelt


@gtx.field_operator
def _T_from_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
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
        (
            ThermodynamicConstants.cvd * (wpfloat(1.0) - qtot)
            + ThermodynamicConstants.cvv * qv
            + ThermodynamicConstants.clw * qliq
            + ThermodynamicConstants.ci * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (
        u + rho * dz * (qliq * ThermodynamicConstants.lvc + qice * ThermodynamicConstants.lsc)
    ) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
    u: fa.CellKField[ta.wpfloat],  # Internal energy (extensive)
    qv: fa.CellKField[ta.wpfloat],  # Water vapor specific humidity
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    temperature: fa.CellKField[ta.wpfloat],  # output
):
    _T_from_internal_energy(u=u, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=temperature)


@gtx.field_operator
def _T_from_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
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
        (
            ThermodynamicConstants.cvd * (wpfloat(1.0) - qtot)
            + ThermodynamicConstants.cvv * qv
            + ThermodynamicConstants.clw * qliq
            + ThermodynamicConstants.ci * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (
        u + rho * dz * (qliq * ThermodynamicConstants.lvc + qice * ThermodynamicConstants.lsc)
    ) / cv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
    u: ta.wpfloat,  # Internal energy (extensive)
    qv: ta.wpfloat,  # Water vapor specific humidity
    qliq: ta.wpfloat,  # Specific mass of liquid phases
    qice: ta.wpfloat,  # Specific mass of solid phases
    rho: ta.wpfloat,  # Ambient density
    dz: ta.wpfloat,  # Extent of grid cell
    temperature: ta.wpfloat,  # output
):
    _T_from_internal_energy_scalar(
        u=u, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=temperature
    )


@gtx.field_operator
def _internal_energy(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
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
    cv = (
        ThermodynamicConstants.cvd * (wpfloat(1.0) - qtot)
        + ThermodynamicConstants.cvv * qv
        + ThermodynamicConstants.clw * qliq
        + ThermodynamicConstants.ci * qice
    )

    return (
        rho * dz * (cv * t - qliq * ThermodynamicConstants.lvc - qice * ThermodynamicConstants.lsc)
    )


@gtx.field_operator
def _internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
    t: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
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
    cv = (
        ThermodynamicConstants.cvd * (wpfloat(1.0) - qtot)
        + ThermodynamicConstants.cvv * qv
        + ThermodynamicConstants.clw * qliq
        + ThermodynamicConstants.ci * qice
    )

    return (
        rho * dz * (cv * t - qliq * ThermodynamicConstants.lvc - qice * ThermodynamicConstants.lsc)
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def internal_energy(  # noqa: PLR0917 [too-many-positional-arguments] GT4Py operator parameters are positional
    t: fa.CellKField[ta.wpfloat],  # Temperature
    qv: fa.CellKField[ta.wpfloat],  # Specific mass of vapor
    qliq: fa.CellKField[ta.wpfloat],  # Specific mass of liquid phases
    qice: fa.CellKField[ta.wpfloat],  # Specific mass of solid phases
    rho: fa.CellKField[ta.wpfloat],  # Ambient density
    dz: fa.CellKField[ta.wpfloat],  # Extent of grid cell
    energy: fa.CellKField[ta.wpfloat],  # output
):
    _internal_energy(t=t, qv=qv, qliq=qliq, qice=qice, rho=rho, dz=dz, out=energy)
