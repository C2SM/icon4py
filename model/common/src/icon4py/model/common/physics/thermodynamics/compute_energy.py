# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_internal_energy_per_area(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the internal energy per unit area from the temperature

    Args:
        t:                 Temperature
        qv:                Specific mass of vapor
        qliq:              Specific mass of liquid phases
        qice:              Specific mass of solid phases
        rho:               Ambient density
        dz:                Vertical extent of grid cell

    Result:                Internal energy per unit area
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
def compute_internal_energy_per_area_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
    t: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the internal energy per unit area from the temperature

    Args:
        t:                 Temperature
        qv:                Specific mass of vapor
        qliq:              Specific mass of liquid phases
        qice:              Specific mass of solid phases
        rho:               Ambient density
        dz:                Vertical extent of grid cell

    Result:                Internal energy per unit area
    """
    qtot = qliq + qice + qv
    cv = (
        PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
        + PhysicsConstants.cvv * qv
        + PhysicsConstants.cpl * qliq
        + PhysicsConstants.cpi * qice
    )

    return rho * dz * (cv * t - qliq * PhysicsConstants.lvc - qice * PhysicsConstants.lsc)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _compute_dry_static_energy(
    temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the dry static energy at full-level cell centers.

    Args:
        temperature: air temperature at full levels [K]
        height_above_ground: height of the full levels above the surface [m]
        grav: gravitational acceleration [m/s2]

    Returns:
        static energy at full levels [J/kg]
    """
    return PhysicsConstants.cpd * temperature + grav * height_above_ground


@gtx.field_operator
def _compute_isochoric_heat_capacity_of_moist_air(  # noqa: PLR0917 [too-many-positional-arguments]
    air_mass: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the heat capacity at constant volume of a mass of moist air.

    Port of get_cvair in mo_aes_phy_diag.f90.

    Args:
        air_mass: mass of moist air, per unit area for ICON's mair [kg m-2]
        qv: specific mass of water vapor [kg/kg]
        qc: specific mass of cloud water [kg/kg]
        qi: specific mass of cloud ice [kg/kg]
        qr: specific mass of rain [kg/kg]
        qs: specific mass of snow [kg/kg]
        qg: specific mass of graupel [kg/kg]

    Returns:
        heat capacity at constant volume, per unit area for ICON's cvair [J K-1 m-2]
    """
    qliq = qc + qr
    qice = qi + qs + qg
    qtot = qv + qice + qliq
    cv = (
        PhysicsConstants.cvd * (wpfloat(1.0) - qtot)
        + PhysicsConstants.cvv * qv
        + PhysicsConstants.cpl * qliq
        + PhysicsConstants.cpi_ref * qice
    )
    return cv * air_mass


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_isochoric_heat_capacity_of_moist_air(  # noqa: PLR0917 [too-many-positional-arguments]
    air_mass: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    heat_capacity: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_isochoric_heat_capacity_of_moist_air(
        air_mass=air_mass,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        out=heat_capacity,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
