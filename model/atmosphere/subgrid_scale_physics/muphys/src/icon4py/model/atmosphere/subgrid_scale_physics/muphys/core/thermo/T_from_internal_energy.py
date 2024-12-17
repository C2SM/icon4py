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
def _T_from_internal_energy(
    u:         fa.CellField[ta.wpfloat],             # Internal energy (extensive)
    qv:        fa.CellField[ta.wpfloat],             # Water vapor specific humidity
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
) -> fa.CellField[ta.wpfloat]:                       # Temperature
    qtot = qliq + qice + qv                          # total water specific mass
    cv   = ( CVD * ( 1.0 - qtot ) + CVV * qv + CLW * qliq + CI * qice ) * rho * dz # Moist isometric specific heat

    return ( u + rho * dz * ( qliq * LVC + qice * LSC )) / cv

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def T_from_internal_energy(
    u:         fa.CellField[ta.wpfloat],             # Internal energy (extensive)
    qv:        fa.CellField[ta.wpfloat],             # Water vapor specific humidity
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
    temperature: fa.CellField[ta.wpfloat]            # output
):
    _T_from_internal_energy( u, qv, qliq, qice, rho, dz, CI, CLW, CVD, CVV, LSC, LVC, out=temperature )
