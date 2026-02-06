# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next import exp, log, neighbor_sum

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.states import static_coefficients as static_coeff, tracer_state
from icon4py.model.common.type_alias import wpfloat


physics_constants: Final = phy_const.PhysicsConstants()


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
