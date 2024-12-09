# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _snow_to_graupel(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,
    TFRZ_HOM: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                      # Return: Riming snow rate
    A_RIM_CT = 0.5                                  # Constants in riming formula
    B_RIM_CT = 0.75
    return where( (minimum(qc,qs) > QMIN) & (t > TFRZ_HOM), A_RIM_CT * qc * power(qs*rho, B_RIM_CT), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_to_graupel(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qs:       fa.CellField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,
    TFRZ_HOM: ta.wpfloat,
    conversion_rate: fa.CellField[ta.wpfloat],      # output
):
    _snow_to_graupel(t, rho, qc, qs, QMIN, TFRZ_HOM, out=conversion_rate)
