# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _ice_deposition_nucleation(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    qc:        fa.CellField[ta.wpfloat],             # Specific humidity of cloud
    qi:        fa.CellField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellField[ta.wpfloat],             # Ice crystal number
    dvsi:      fa.CellField[ta.wpfloat],             # Vapor excess with respect to ice sat
    dt:        ta.wpfloat,                           # Time step
    QMIN:      ta.wpfloat,
    M0_ICE:    ta.wpfloat,
    TFRZ_HET1: ta.wpfloat,
    TFRZ_HET2: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Rate of vapor deposition for new ice
    return where( qi <= QMIN & ((t < TFRZ_HET2) & (dvsi > 0.0)) | (t <= TFRZ_HET1 & qc > QMIN), minimum(M0_ICE * ni, maximum(0.0, dvsi)) / dt, 0.0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def ice_deposition_nucleation(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    qc:        fa.CellField[ta.wpfloat],             # Specific humidity of cloud
    qi:	       fa.CellField[ta.wpfloat],             # Specific humidity of ice
    ni:	       fa.CellField[ta.wpfloat],             # Ice crystal number
    dvsi:      fa.CellField[ta.wpfloat],             # Vapor excess with respect to ice sat
    dt:	       ta.wpfloat,                           # Time step 
    QMIN:      ta.wpfloat,
    M0_ICE:    ta.wpfloat,
    TFRZ_HET1: ta.wpfloat,
    TFRZ_HET2: ta.wpfloat,
    vapor_deposition_rate: fa.CellField[ta.wpfloat]  # output
):
    _ice_deposition_nucleation( t, qc, qi, ni, dvsi, dt, QMIN, M0_ICE, TFRZ_HET1, TFRZ_HET2, out=vapor_deposition_rate )
