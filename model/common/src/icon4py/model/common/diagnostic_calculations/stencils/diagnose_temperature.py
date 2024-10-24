# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _diagnose_virtual_temperature_and_temperature(
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    rv_o_rd_minus1: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    qsum = qc + qi + qr + qs + qg
    virtual_temperature = theta_v * exner
    temperature = virtual_temperature / (1.0 + rv_o_rd_minus1 * qv - qsum)
    return virtual_temperature, temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def diagnose_virtual_temperature_and_temperature(
    qv: fa.CellKField[ta.wpfloat],
    # TODO (Chia Rui): This should be changed to a list hydrometeors with mass instead of directly specifying each hydrometeor, as in trHydroMass list in ICON. Otherwise, the input arguments may need to be changed when different microphysics is used.
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    rv_o_rd_minus1: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
