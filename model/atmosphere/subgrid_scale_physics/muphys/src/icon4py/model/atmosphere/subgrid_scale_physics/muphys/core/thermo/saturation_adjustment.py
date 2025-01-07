# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _saturation_adjustment(
    te:        fa.CellField[ta.wpfloat],             # Temperature
    qve:       fa.CellField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellField[ta.wpfloat],             # Specific rain water
    gti:       fa.CellField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    rho:       fa.CellField[ta.wpfloat],             # Density containing dry air and water constituents
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Internal energy
    return rho * dz * ( cv * t - qliq * LVC - qice * LSC )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def saturation_adjustment(
    te:        fa.CellField[ta.wpfloat],             # Temperature
    qve:       fa.CellField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellField[ta.wpfloat],             # Specific rain water
    gti:       fa.CellField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    rho:       fa.CellField[ta.wpfloat],             # Density containing dry air and water constituents
    internal_energy: fa.CellField[ta.wpfloat]  # output
):
    _internal_energy( t, qv, qliq, qice, rho, dz, CI, CLW, CVD, CVV, LSC, LVC, out=internal_energy )
