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
def _cloud_to_graupel(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qg:       fa.CellField[ta.wpfloat],             # Graupel specific mass
    TFRZ_HOM: ta.wpfloat,
    QMIN:     ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                     # Return: Riming graupel rate
    A_RIM = 4.43
    B_RIM = 0.94878
    return where( (minimum(qc,qg) > QMIN) & (t > TFRZ_HOM), A_RIM * qc * power(qg * rho, B_RIM), 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_graupel(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    rho:      fa.CellField[ta.wpfloat],             # Ambient density
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qg:       fa.CellField[ta.wpfloat],             # Graupel specific mass
    TFRZ_HOM:                ta.wpfloat,
    QMIN:                    ta.wpfloat,
    riming_graupel_rate:     fa.CellField[ta.wpfloat],             # output
):
    _cloud_to_graupel(t, rho, qc, qg, TFRZ_HOM, QMIN, out=riming_graupel_rate)
