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
def _internal_energy(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    qv:        fa.CellField[ta.wpfloat],             # Specific mass of vapor
    qliq:      fa.CellField[ta.wpfloat],             # Specific mass of liquid phases
    qice:      fa.CellField[ta.wpfloat],             # Specific mass of solid phases
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    dz:        fa.CellField[ta.wpfloat],             # Extent of grid cell
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LSC:       ta.wpfloat,
    LVC:       ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                       # Internal energy
    qtot = qliq + qice + qv
    cv   = CVD * ( 1.0 - qtot ) + CVV * qv + CLW * qliq + CI * qice

    return rho * dz * ( cv * t - qliq * LVC - qice * LSC )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def internal_energy(
    t:         fa.CellField[ta.wpfloat],             # Temperature
    qv:        fa.CellField[ta.wpfloat],             # Specific mass of vapor
    qliq:      fa.CellField[ta.wpfloat],             # Specific mass of liquid phases
    qice:      fa.CellField[ta.wpfloat],             # Specific mass of solid phases
    rho:       fa.CellField[ta.wpfloat],             # Ambient density
    dz:        fa.CellField[ta.wpfloat],             # Extent of grid cell
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CV:        ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LSC:       ta.wpfloat,
    LVC:       ta.wpfloat,
    internal_energy: fa.CellField[ta.wpfloat]  # output
):
    _internal_energy( t, qv, qliq, qice, rho, dz, CI, CLW, CV, CVD, CVV, LSC, LVC, out=internal_energy )
