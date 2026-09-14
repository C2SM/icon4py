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


@gtx.field_operator
def _compute_dry_static_energy(
    temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
) -> fa.CellKField[wpfloat]:
    """
    Compute the dry static energy at full-level cell centers.

    Port of ``compute_static_energy`` in ICON's ``mo_vdf_atmo.f90``:

        dry_static_energy = cpd * temperature + grav * height_above_ground

    ``height_above_ground`` is the height of the full levels above the surface
    (``ghf``), so ``grav * height_above_ground`` is the geopotential above ground.

    Args:
        temperature: air temperature at full levels [K]
        height_above_ground: height of the full levels above the surface [m]
        grav: gravitational acceleration [m/s2]

    Returns:
        static energy at full levels [J/kg]
    """
    return PhysicsConstants.cpd * temperature + grav * height_above_ground


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_dry_static_energy(  # noqa: PLR0917 [too-many-positional-arguments]
    temperature: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    dry_static_energy: fa.CellKField[wpfloat],
    grav: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_dry_static_energy(
        temperature=temperature,
        height_above_ground=height_above_ground,
        grav=grav,
        out=dry_static_energy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
