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


@gtx.field_operator
def _diagnose_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    surface_pressure = PhysicsConstants.p0ref * exp(
        PhysicsConstants.cpd_o_rd * log(exner(dims.KHalfDim - 2.5))
        + PhysicsConstants.grav_o_rd
        * (
            ddqz_z_full(dims.KHalfDim - 0.5) / virtual_temperature(dims.KHalfDim - 0.5)
            + ddqz_z_full(dims.KHalfDim - 1.5) / virtual_temperature(dims.KHalfDim - 1.5)
            + 0.5 * ddqz_z_full(dims.KHalfDim - 2.5) / virtual_temperature(dims.KHalfDim - 2.5)
        )
    )
    return surface_pressure


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_surface_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellKHalfField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _diagnose_surface_pressure(
        exner=exner,
        virtual_temperature=virtual_temperature,
        ddqz_z_full=ddqz_z_full,
        out=surface_pressure,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_start, vertical_end),
        },
    )
