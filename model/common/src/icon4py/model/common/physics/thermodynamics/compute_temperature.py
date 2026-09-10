# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import exp, log

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_virtual_temperature_and_temperature(  # noqa: PLR0917 [too-many-positional-arguments]
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    qsum = qc + qi + qr + qs + qg
    virtual_temperature = theta_v * exner
    temperature = virtual_temperature / (1.0 + PhysicsConstants.rv_o_rd_minus_1 * qv - qsum)
    return virtual_temperature, temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_virtual_temperature_and_temperature(  # noqa: PLR0917 [too-many-positional-arguments]
    qv: fa.CellKField[ta.wpfloat],
    # TODO(OngChia): This should be changed to a list hydrometeors with mass instead of directly specifying each hydrometeor, as in trHydroMass list in ICON. Otherwise, the input arguments may need to be changed when different microphysics is used.
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_virtual_temperature_and_temperature(
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        theta_v=theta_v,
        exner=exner,
        out=(virtual_temperature, temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_exner_and_theta_v(
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """Update exner and theta_v from a physics virtual-temperature tendency.

    Recompute exner from the new virtual temperature via the exact equation of
    state and diagnose ``theta_v = Tv / exner``, so the exner/rho/theta_v trio
    stays EOS-consistent::

        Tv_new = Tv + dtime * dTv / dt
        exner_new = (rd / p0ref * rho * Tv_new) ** (rd / cpd)
        theta_v = Tv_new / exner_new

    Args:
        rho: air density [kg m-3]
        virtual_temperature: virtual temperature before the physics update [K]
        virtual_temperature_tendency: physics virtual-temperature tendency [K s-1]
        dtime: time step [s]
    Returns:
        (new exner function, new virtual potential temperature theta_v [K])
    """
    new_virtual_temperature = virtual_temperature + virtual_temperature_tendency * dtime
    new_exner = exp(
        PhysicsConstants.rd_o_cpd * log(PhysicsConstants.rd_o_p0ref * rho * new_virtual_temperature)
    )
    new_theta_v = new_virtual_temperature / new_exner
    return new_exner, new_theta_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_exner_and_theta_v(  # noqa: PLR0917 [too-many-positional-arguments]
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_exner_and_theta_v(
        rho=rho,
        virtual_temperature=virtual_temperature,
        virtual_temperature_tendency=virtual_temperature_tendency,
        dtime=dtime,
        out=(exner, theta_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_temperature_from_internal_energy_per_area(  # noqa: PLR0917 [too-many-positional-arguments]
    internal_energy_per_area: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qliq: fa.CellKField[ta.wpfloat],
    qice: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dz: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the temperature from the internal energy per unit area

    Args:
        internal_energy_per_area:  Internal energy per unit area
        qv:                        Water vapor specific humidity
        qliq:                      Specific mass of liquid phases
        qice:                      Specific mass of solid phases
        rho:                       Ambient density
        dz:                        Vertical extent of grid cell

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
    )  # Moist heat capacity per unit area

    return (
        internal_energy_per_area
        + rho * dz * (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)
    ) / cv


@gtx.field_operator
def compute_temperature_from_internal_energy_per_area_scalar(  # noqa: PLR0917 [too-many-positional-arguments]
    internal_energy_per_area: ta.wpfloat,
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
) -> ta.wpfloat:
    """
    Compute the temperature from the internal energy per unit area (scalar version callable from scan_operator)

    Args:
        internal_energy_per_area:  Internal energy per unit area
        qv:                        Water vapor specific humidity
        qliq:                      Specific mass of liquid phases
        qice:                      Specific mass of solid phases
        rho:                       Ambient density
        dz:                        Vertical extent of grid cell

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
    )  # Moist heat capacity per unit area

    return (
        internal_energy_per_area
        + rho * dz * (qliq * PhysicsConstants.lvc + qice * PhysicsConstants.lsc)
    ) / cv
