# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@field_operator
def _diagnose_virtual_temperature_and_temperature(
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rv_o_rd_minus1: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    qsum = qc + qi + qr + qs + qg
    virtual_temperature = theta_v * exner
    temperature = virtual_temperature / (1.0 + rv_o_rd_minus1 * qv - qsum)
    return virtual_temperature, temperature


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def diagnose_virtual_temperature_and_temperature(
    qv: fa.CellKField[wpfloat],
    # TODO (Chia Rui): This should be changed to a list hydrometeors with mass instead of directly specifying each hydrometeor, as in trHydroMass list in ICON. Otherwise, the input arguments may need to be changed when different microphysics is used.
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    rv_o_rd_minus1: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _diagnose_virtual_temperature_and_temperature(
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        theta_v,
        exner,
        rv_o_rd_minus1,
        out=(virtual_temperature, temperature),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
