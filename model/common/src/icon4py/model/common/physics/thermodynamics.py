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

The constants come from :class:`icon4py.model.common.constants.PhysicsConstants`
(a ``wpfloat``-based ``enum.Enum``, which is what GT4Py/gtfn needs for symbols
referenced inside field operators).
"""

import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def compute_temperature_from_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
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
            PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
            + PhysicsConstants.cvv * qv
            + PhysicsConstants.cpl * qliq
            + PhysicsConstants.cpi * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)) / cv


@gtx.field_operator
def compute_temperature_from_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
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
            PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
            + PhysicsConstants.cvv * qv
            + PhysicsConstants.cpl * qliq
            + PhysicsConstants.cpi * qice
        )
        * rho
        * dz
    )  # Moist isometric specific heat

    return (u + rho * dz * (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)) / cv


@gtx.field_operator
def compute_internal_energy(  # noqa: PLR0917 [too-many-positional-arguments]
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
        PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
        + PhysicsConstants.cvv * qv
        + PhysicsConstants.cpl * qliq
        + PhysicsConstants.cpi * qice
    )

    return rho * dz * (cv * t - qliq * PhysicsConstants.lvc - qice * PhysicsConstants.lsc)


@gtx.field_operator
def compute_internal_energy_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
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
        PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
        + PhysicsConstants.cvv * qv
        + PhysicsConstants.cpl * qliq
        + PhysicsConstants.cpi * qice
    )

    return rho * dz * (cv * t - qliq * PhysicsConstants.lvc - qice * PhysicsConstants.lsc)
