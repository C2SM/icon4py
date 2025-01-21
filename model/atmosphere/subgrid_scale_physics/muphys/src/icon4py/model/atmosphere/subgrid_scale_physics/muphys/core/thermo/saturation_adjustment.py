# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.qsat_rho import _qsat_rho
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.dqsatdT_rho import _dqsatdT_rho

@gtx.field_operator
def _newton_raphson(
    Tx: fa.CellField[ta.wpfloat],
    rho: fa.CellField[ta.wpfloat],
    qve: fa.CellField[ta.wpfloat],
    qce: fa.CellField[ta.wpfloat],
    cvc: fa.CellField[ta.wpfloat],
    ue: fa.CellField[ta.wpfloat],
    CVV:   ta.wpfloat,
    CLW:   ta.wpfloat,
    LVC:   ta.wpfloat,
    TMELT: ta.wpfloat,
    RV:    ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:
    qx  = _qsat_rho(Tx, rho, TMELT, RV)
    dqx = _dqsatdT_rho(qx, Tx, TMELT)
    qcx = qve + qce - qx
    cv  = cvc + CVV * qx + CLW * qcx
    ux  = cv * Tx - qcx * LVC
    dux = cv + dqx * (LVC + (CVV - CLW) * Tx)
    Tx  = Tx - (ux - ue) / dux
    return Tx

@gtx.field_operator
def _saturation_adjustment(
    te:        fa.CellField[ta.wpfloat],             # Temperature
    qve:       fa.CellField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellField[ta.wpfloat],             # Specific rain water
    qti:       fa.CellField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    rho:       fa.CellField[ta.wpfloat],             # Density containing dry air and water constituents
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LVC:       ta.wpfloat,
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
) -> tuple[fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat],fa.CellField[bool]]:                       # Internal energy

    qt = qve + qce + qre + qti
    cvc = CVD * (1.0-qt) + CLW * qre + CI * qti
    cv = cvc + CVV * qve + CLW * qce
    ue = cv * te - qce * LVC
    Tx_hold = ue / (cv + qce * (CVV - CLW))
    qx_hold = _qsat_rho(Tx_hold, rho, TMELT, RV)

    Tx = te
    # Newton-Raphson iteration: 6 times the same operations
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV)

    # At this point we hope Tx has converged
    qx = _qsat_rho(Tx, rho, TMELT, RV)

    # Is it possible to unify the where for all three outputs??
    mask = ( qve+qce <= qx_hold )
    te  = where( ( qve+qce <= qx_hold ), Tx_hold, Tx )
    qce = where( ( qve+qce <= qx_hold ), 0.0, maximum(qve+qce-qx, 0.0) )
    qve = where( ( qve+qce <= qx_hold ), qve+qce, qx )

    return te, qve, qce, mask

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def saturation_adjustment(
    te:        fa.CellField[ta.wpfloat],             # Temperature
    qve:       fa.CellField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellField[ta.wpfloat],             # Specific rain water
    qti:       fa.CellField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    rho:       fa.CellField[ta.wpfloat],             # Density containing dry air and water constituents
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LVC:       ta.wpfloat,
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
    te_out:    fa.CellField[ta.wpfloat],             # Temperature
    qve_out:   fa.CellField[ta.wpfloat],             # Specific humidity
    qce_out:   fa.CellField[ta.wpfloat],             # Specific cloud water content
    mask_out:  fa.CellField[bool]                    # Specific cloud water content
):
    _saturation_adjustment( te, qve, qce, qre, qti, rho, CI, CLW, CVD, CVV, LVC, TMELT, RV, out=(te_out, qve_out, qce_out, mask_out) )
