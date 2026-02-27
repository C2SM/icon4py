# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import exp, sqrt, log, neighbor_sum

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import Koff, E2C, E2CDim
from icon4py.model.common.states import static_coefficients as static_coeff, tracer_state
from icon4py.model.common.type_alias import wpfloat


physics_constants: Final = phy_const.PhysicsConstants()


@gtx.scan_operator(axis=dims.KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[ta.wpfloat, ta.wpfloat, bool],
    ddqz_z_full: ta.wpfloat,
    virtual_temperature: ta.wpfloat,
    surface_pressure: ta.wpfloat,
):
    pressure_interface = (
        surface_pressure * exp(-physics_constants.grav_o_rd * ddqz_z_full / virtual_temperature)
        if state[2]
        else state[1] * exp(-physics_constants.grav_o_rd * ddqz_z_full / virtual_temperature)
    )
    pressure = (
        sqrt(surface_pressure * pressure_interface)
        if state[2]
        else sqrt(state[1] * pressure_interface)
    )
    return pressure, pressure_interface, False


@gtx.field_operator
def _diagnose_pressure(
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Update pressure by assuming hydrostatic balance (dp/dz = -rho g = p g / Rd / Tv).
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        surface_pressure: surface air pressure [Pa]
    Returns:
        pressure at full levels, pressure at half levels (excluding surface level)
    """
    pressure, pressure_ifc, _ = _scan_pressure(ddqz_z_full, virtual_temperature, surface_pressure)
    return pressure, pressure_ifc


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_pressure(
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    pressure_ifc: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_pressure(
        ddqz_z_full,
        virtual_temperature,
        surface_pressure,
        out=(pressure, pressure_ifc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _diagnose_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    surface_pressure = physics_constants.p0ref * exp(
        physics_constants.cpd_o_rd * log(exner(Koff[-3]))
        + physics_constants.grav_o_rd
        * (
            ddqz_z_full(Koff[-1]) / virtual_temperature(Koff[-1])
            + ddqz_z_full(Koff[-2]) / virtual_temperature(Koff[-2])
            + 0.5 * ddqz_z_full(Koff[-3]) / virtual_temperature(Koff[-3])
        )
    )
    return surface_pressure


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_surface_pressure(
        exner,
        virtual_temperature,
        ddqz_z_full,
        out=surface_pressure,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _total_hydrometeors(
    tracers: tracer_state.TracerState,
) -> fa.CellKField[ta.wpfloat]:
    qsum = tracers.qc + tracers.qi + tracers.qr + tracers.qs + tracers.qg
    return qsum


@gtx.field_operator
def _diagnose_temperature(
    tracers: tracer_state.TracerState,
    virtual_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    temperature = virtual_temperature / (
        wpfloat("1.0")
        + physics_constants.rv_o_rd_minus_1 * tracers.qv
        - _total_hydrometeors(tracers)
    )
    return temperature


@gtx.field_operator
def _diagnose_virtual_temperature(
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    virtual_temperature = temperature * (
        wpfloat("1.0")
        + physics_constants.rv_o_rd_minus_1 * tracers.qv
        - _total_hydrometeors(tracers)
    )
    return virtual_temperature, temperature


@gtx.field_operator
def _diagnose_virtual_temperature_and_temperature_from_exner(
    tracers: tracer_state.TracerState,
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    virtual_temperature = theta_v * exner
    temperature = _diagnose_temperature(tracers, virtual_temperature)
    return virtual_temperature, temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_virtual_temperature_and_temperature_from_exner(
    tracers: tracer_state.TracerState,
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_virtual_temperature_and_temperature_from_exner(
        tracers,
        theta_v,
        exner,
        out=(virtual_temperature, temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _diagnose_exner_from_virtual_temperature_and_rho(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    exner = exp(
        physics_constants.rd_o_cpd * log(physics_constants.rd_o_p0ref * rho * virtual_temperature)
    )
    return exner


@gtx.field_operator
def _diagnose_exner_from_virtual_temperature(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    old_virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    exner = exner * (
        wpfloat("1.0")
        + physics_constants.rd_o_cpd
        * (virtual_temperature / old_virtual_temperature - wpfloat("1.0"))
    )
    return exner


@gtx.field_operator
def _diagnose_theta_v(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    theta_v = virtual_temperature / exner
    return theta_v


@gtx.field_operator
def _diagnose_exner_and_theta_v_from_virtual_temperature(
    perturbed_exner: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    previous_exner: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    virtual_temperature = _diagnose_virtual_temperature(tracers, temperature)
    exner = _diagnose_exner_from_virtual_temperature_and_rho(virtual_temperature, rho)
    perturbed_exner = perturbed_exner + exner - previous_exner
    theta_v = _diagnose_theta_v(virtual_temperature, exner)
    return virtual_temperature, exner, perturbed_exner, theta_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_exner_and_theta_v_from_virtual_temperature(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    perturbed_exner: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    previous_exner: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_exner_and_theta_v_from_virtual_temperature(
        perturbed_exner,
        tracers,
        temperature,
        rho,
        previous_exner,
        out=(virtual_temperature, exner, perturbed_exner, theta_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _diagnose_exner_and_virtual_temperature(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    new_virtual_temperature = _diagnose_virtual_temperature(tracers, temperature)
    new_exner = _diagnose_exner_from_virtual_temperature(
        new_virtual_temperature, virtual_temperature, exner
    )
    return new_virtual_temperature, new_exner


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_exner_and_virtual_temperature(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_exner_and_virtual_temperature(
        virtual_temperature,
        exner,
        tracers,
        temperature,
        out=(virtual_temperature, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_vn_from_u_v_tendencies(
    vn: fa.EdgeKField[ta.wpfloat],
    u_tendency: fa.CellKField[ta.wpfloat],
    v_tendency: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    coeff: static_coeff.StaticCoeff,
) -> fa.EdgeKField[ta.wpfloat]:
    new_vn = vn + dt * neighbor_sum(
        coeff.c_lin_e
        * (
            u_tendency(E2C) * coeff.primal_normal_cell_x
            + v_tendency(E2C) * coeff.primal_normal_cell_y
        ),
        axis=E2CDim,
    )
    return new_vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_vn_from_u_v_tendencies(
    vn: fa.EdgeKField[ta.wpfloat],
    u_tendency: fa.CellKField[ta.wpfloat],
    v_tendency: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    coeff: static_coeff.StaticCoeff,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_vn_from_u_v_tendencies(
        vn,
        u_tendency,
        v_tendency,
        dt,
        coeff,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _calculate_virtual_temperature_tendency(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update virtual temperature tendency.

    Args:
        dtime: time step [s]
        qv: specific humidity [kg kg-1]
        qc: specific cloud water content [kg kg-1]
        qi: specific cloud ice content [kg kg-1]
        qr: specific rain water content [kg kg-1]
        qs: specific snow content [kg kg-1]
        qg: specific graupel content [kg kg-1]
        temperature: air temperature [K]
        virtual_temperature: air virtual temperature [K]
    Returns:
        virtual temperature tendency [K s-1], exner tendency [s-1], new exner, new virtual temperature [K]
    """
    qsum = qc + qi + qr + qs + qg

    new_virtual_temperature = temperature * (
        wpfloat("1.0") + physics_constants.rv_o_rd_minus_1 * qv - qsum
    )

    return (new_virtual_temperature - virtual_temperature) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_virtual_temperature_tendency(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_virtual_temperature_tendency(
        dtime,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        temperature,
        virtual_temperature,
        out=virtual_temperature_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _calculate_exner_tendency(
    dtime: ta.wpfloat,
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update exner tendency.

    Args:
        dtime: time step [s]
        virtual_temperature: air virtual temperature [K]
        virtual_temperature_tendency: air virtual temperature tendency [K s-1]
        exner: exner function
    Returns:
        exner tendency [s-1]
    """

    new_exner = exner * (
        wpfloat("1.0")
        + physics_constants.rd_o_cpd * virtual_temperature_tendency / virtual_temperature * dtime
    )

    return (new_exner - exner) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_exner_tendency(
    dtime: ta.wpfloat,
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    exner_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_exner_tendency(
        dtime,
        virtual_temperature,
        virtual_temperature_tendency,
        exner,
        out=exner_tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _calculate_cell_kdim_field_tendency(
    dtime: ta.wpfloat,
    old_field: fa.CellKField[ta.wpfloat],
    new_field: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update tendency of Cell-K Dim field.

    Args:
        dtime: time step [s]
        old_field: any old Cell-K Dim field [unit]
        new_field: any new Cell-K Dim field [unit]
    Returns:
        tendency [unit s-1]
    """

    return (new_field - old_field) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_cell_kdim_field_tendency(
    dtime: ta.wpfloat,
    old_field: fa.CellKField[ta.wpfloat],
    new_field: fa.CellKField[ta.wpfloat],
    tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_cell_kdim_field_tendency(
        dtime,
        old_field,
        new_field,
        out=tendency,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
