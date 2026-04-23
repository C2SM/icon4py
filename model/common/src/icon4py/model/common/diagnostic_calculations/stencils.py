# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import exp, log, neighbor_sum, sqrt, where

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import E2C, E2CDim, Koff
from icon4py.model.common.states import tracer_state
from icon4py.model.common.type_alias import wpfloat


physics_constants: Final = phy_const.PhysicsConstants()


@gtx.field_operator
def _diagnose_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
        Diagnose surface pressure by assuming hydrostatic balance (dp/dz = -rho g = - p g / Rd / Tv).
        Note that virtual temperature is used in the equation to include the moist effect.

    Args:
        exner: exner function
        virtual_temperature): virtual temperature [K]
        ddqz_z_full: vertical grid spacing at full levels [m]

    Returns:
        surface pressure: air pressure on the surface (model bottom boundary) [Pa]
    """
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
    surface_pressure: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
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


@gtx.scan_operator(axis=dims.KDim, forward=False, init=(0.0, 0.0, True))
def _scan_pressure(
    state: tuple[ta.wpfloat, ta.wpfloat, bool],
    ddqz_z_full: ta.wpfloat,
    virtual_temperature: ta.wpfloat,
    surface_pressure: ta.wpfloat,
) -> tuple[ta.wpfloat, ta.wpfloat, bool]:
    """
        Diagnose pressure at the model full and half levels by assuming hydrostatic balance (dp/dz = -rho g = - p g / Rd / Tv).
        Note that virtual temperature is used in the equation to include the moist effect.
        The hydrostatic balance is integrated from half levels k-1/2 to k+1/2, and we can obtain the pressure at k+1/2 half level given the pressure at k-1/2 half level.
        The pressure at full level k is diagnosed by assuming the geometric mean of the pressure at two adjacent half levels.

    Args:
        state: a tuple of (pressure at full levels, pressure at half levels, switch), where switch is True when the current level is the bottommost model level (scan from bottom to top) [Pa]
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: virtual temperature [K]
        surface_pressure: air pressure on the surface (model bottom boundary) [Pa]

    Returns:
        pressure at full levels, pressure at half levels [Pa]
    """
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
    surface_pressure: gtx.Field[gtx.Dims[dims.CellDim], ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Update pressure by assuming hydrostatic balance (dp/dz = -rho g = - p g / Rd / Tv).
    Note that virtual temperature is used in the equation.

    Args:
        ddqz_z_full: vertical grid spacing at full levels [m]
        virtual_temperature: air virtual temperature [K]
        surface_pressure: air pressure on the surface (model bottom boundary) [Pa]
    Returns:
        pressure at full levels, pressure at half levels (excluding surface level) [Pa]
    """
    pressure, pressure_at_half_levels, _ = _scan_pressure(
        ddqz_z_full, virtual_temperature, surface_pressure
    )
    return pressure, pressure_at_half_levels


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_pressure(
    pressure: fa.CellKField[ta.wpfloat],
    pressure_at_half_levels: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_pressure(
        surface_pressure,
        virtual_temperature,
        ddqz_z_full,
        out=(pressure, pressure_at_half_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _total_hydrometeors(
    tracers: tracer_state.TracerState,
) -> fa.CellKField[ta.wpfloat]:
    """
    Summation of all hydrometeor mixing ratios.

    Args:
        tracers: tracer state containing the mixing ratios of water vapor and all hydrometeors [kg kg-1]
    Returns:
        total hydrometeor mixing ratio [kg kg-1]
    """
    qsum = tracers.qc + tracers.qi + tracers.qr + tracers.qs + tracers.qg
    return qsum


@gtx.field_operator
def _diagnose_temperature(
    tracers: tracer_state.TracerState,
    virtual_temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Diagnose temperature .

    Args:
        tracers: tracer state containing the mixing ratios of water vapor and all hydrometeors [kg kg-1]
    Returns:
        total hydrometeor mixing ratio [kg kg-1]
    """
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
    virtual_temperature: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
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
def _diagnose_virtual_temperature_and_exner(
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
def diagnose_virtual_temperature_and_exner(
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _diagnose_virtual_temperature_and_exner(
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
def _update_exner_and_theta_v_from_virtual_temperature_in_halo(
    exner: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    mask_prog_halo_c: fa.CellField[bool],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    exner = where(
        mask_prog_halo_c,
        _diagnose_exner_from_virtual_temperature_and_rho(virtual_temperature, rho),
        exner,
    )
    theta_v = where(
        mask_prog_halo_c,
        _diagnose_theta_v(virtual_temperature=virtual_temperature, exner=exner),
        theta_v,
    )
    return exner, theta_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_exner_and_theta_v_from_virtual_temperature_in_halo(
    exner: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    mask_prog_halo_c: fa.CellField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_exner_and_theta_v_from_virtual_temperature_in_halo(
        exner,
        theta_v,
        rho,
        virtual_temperature,
        mask_prog_halo_c,
        out=(exner, theta_v),
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
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    new_vn = vn + dt * neighbor_sum(
        c_lin_e * (u_tendency(E2C) * primal_normal_cell_x + v_tendency(E2C) * primal_normal_cell_y),
        axis=E2CDim,
    )
    return new_vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_vn_from_u_v_tendencies(
    vn: fa.EdgeKField[ta.wpfloat],
    u_tendency: fa.CellKField[ta.wpfloat],
    v_tendency: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
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
        c_lin_e,
        primal_normal_cell_x,
        primal_normal_cell_y,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_satad_output_from_tendency(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    temperature_tendency: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Update temperature, qv, and qc from their tendency.

    Args:
        temperature: air temperature [K]
        qv: specific humidity [kg kg-1]
        qc: specific cloud water content [kg kg-1]
        temperature_tendency: temperature tendency [K s-1]
        qv_tendency: specific humidity tendency [s-1]
        qc_tendency: specific cloud water content tendency [s-1]
        dtime: time step [s]
    Returns:
        updated temperature, qv, qc
    """
    return (
        temperature + temperature_tendency * dtime,
        qv + qv_tendency * dtime,
        qc + qc_tendency * dtime,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_satad_output_from_tendency(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    temperature_tendency: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_satad_output_from_tendency(
        temperature,
        qv,
        qc,
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        dtime,
        out=(temperature, qv, qc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_microphysics_output_from_tendency(
    temperature: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature_tendency: fa.CellKField[ta.wpfloat],
    tracer_tendency: tracer_state.TracerStateTendency,
    dtime: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """
    Update temperature and all hydrometeros from their tendency.

    Args:
        temperature: air temperature [K]
        tracers: hydrometeor mixing ratios [kg kg-1]
        temperature_tendency: temperature tendency [K s-1],
        tracer_tendency: tendency of hydrometeor mixing ratios [s-1]
        dtime: time step [s]
    Returns:
        updated temperature, hydrometeor mixing ratios
    """
    return (
        temperature + temperature_tendency * dtime,
        tracers.qv + tracer_tendency.qv_tendency * dtime,
        tracers.qc + tracer_tendency.qc_tendency * dtime,
        tracers.qr + tracer_tendency.qr_tendency * dtime,
        tracers.qi + tracer_tendency.qi_tendency * dtime,
        tracers.qs + tracer_tendency.qs_tendency * dtime,
        tracers.qg + tracer_tendency.qg_tendency * dtime,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_microphysics_output_from_tendency(
    temperature: fa.CellKField[ta.wpfloat],
    tracers: tracer_state.TracerState,
    temperature_tendency: fa.CellKField[ta.wpfloat],
    tracer_tendency: tracer_state.TracerStateTendency,
    dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_microphysics_output_from_tendency(
        temperature,
        tracers,
        temperature_tendency,
        tracer_tendency,
        dtime,
        out=(temperature, tracers.qv, tracers.qc, tracers.qr, tracers.qi, tracers.qs, tracers.qg),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
