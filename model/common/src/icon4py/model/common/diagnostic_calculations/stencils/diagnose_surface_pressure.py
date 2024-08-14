# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import exp, int32, log

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _diagnose_surface_pressure(
    exner: fa.CellKField[vpfloat],
    temperature: fa.CellKField[vpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
) -> fa.CellKField[vpfloat]:
    pressure_sfc = p0ref * exp(
        cpd_o_rd * log(exner(Koff[-3]))
        + grav_o_rd
        * (
            ddqz_z_full(Koff[-1]) / temperature(Koff[-1])
            + ddqz_z_full(Koff[-2]) / temperature(Koff[-2])
            + 0.5 * ddqz_z_full(Koff[-3]) / temperature(Koff[-3])
        )
    )
    return pressure_sfc


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_surface_pressure(
    exner: fa.CellKField[vpfloat],
    temperature: fa.CellKField[vpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    pressure_sfc: fa.CellKField[vpfloat],
    cpd_o_rd: wpfloat,
    p0ref: wpfloat,
    grav_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_surface_pressure(
        exner,
        temperature,
        ddqz_z_full,
        cpd_o_rd,
        p0ref,
        grav_o_rd,
        out=pressure_sfc,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
