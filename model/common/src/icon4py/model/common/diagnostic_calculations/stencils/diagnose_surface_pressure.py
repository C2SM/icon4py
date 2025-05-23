# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import exp, log

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import Koff


physics_constants: Final = phy_const.PhysicsConstants()


@field_operator
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


@program(grid_type=GridType.UNSTRUCTURED)
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
