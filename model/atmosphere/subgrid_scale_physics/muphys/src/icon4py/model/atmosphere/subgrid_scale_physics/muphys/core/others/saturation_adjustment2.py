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
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import _qsat_rho
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import _dqsatdT_rho

@gtx.field_operator
def _satadj_init(
    te:        fa.CellKField[ta.wpfloat],             # Temperature
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellKField[ta.wpfloat],             # Specific rain water
    qti:       fa.CellKField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LVC:       ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]: 
    qt = qve + qce + qre + qti                       # temporary, used only here
    cvc = CVD * (1.0-qt) + CLW * qre + CI * qti      # output variable
    cv = cvc + CVV * qve + CLW * qce                 # temporary, used only here
    ue = cv * te - qce * LVC                         # output variable
    Tx_hold = ue / (cv + qce * (CVV - CLW))
    Tx = te
    return cvc, ue, Tx_hold, Tx                      # output variables

@gtx.field_operator
def _output_calculation(
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qx_hold:   fa.CellKField[ta.wpfloat],             # TBD
    qx:        fa.CellKField[ta.wpfloat],             # TBD
    Tx_hold:   fa.CellKField[ta.wpfloat],             # TBD
    Tx:        fa.CellKField[ta.wpfloat],             # TBD
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:                       # Internal energy

    te  = where( ( qve+qce <= qx_hold ), Tx_hold, Tx )
    qce = where( ( qve+qce <= qx_hold ), 0.0, maximum(qve+qce-qx, 0.0) )
    qve = where( ( qve+qce <= qx_hold ), qve+qce, qx )
    return te, qve, qce

@gtx.field_operator
def _newton_raphson(
    qx:  fa.CellKField[ta.wpfloat],
    dqx: fa.CellKField[ta.wpfloat],
    Tx:  fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qve: fa.CellKField[ta.wpfloat],
    qce: fa.CellKField[ta.wpfloat],
    cvc: fa.CellKField[ta.wpfloat],
    ue:  fa.CellKField[ta.wpfloat],
    CVV:   ta.wpfloat,
    CLW:   ta.wpfloat,
    LVC:   ta.wpfloat,
    TMELT: ta.wpfloat,
    RV:    ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    qcx = qve + qce - qx
    cv  = cvc + CVV * qx + CLW * qcx
    ux  = cv * Tx - qcx * LVC
    dux = cv + dqx * (LVC + (CVV - CLW) * Tx)
    Tx  = Tx - (ux - ue) / dux
    return Tx

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def saturation_adjustment2(
    te:        fa.CellKField[ta.wpfloat],             # Temperature
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellKField[ta.wpfloat],             # Specific rain water
    qti:       fa.CellKField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    rho:       fa.CellKField[ta.wpfloat],             # Density containing dry air and water constituents
    cvc:       fa.CellKField[ta.wpfloat],             # Temporary field
    ue:        fa.CellKField[ta.wpfloat],             # Temporary field
    Tx_hold:   fa.CellKField[ta.wpfloat],             # Temporary field
    Tx:        fa.CellKField[ta.wpfloat],             # Temporary field
    qx_hold:   fa.CellKField[ta.wpfloat],             # Temporary field
    qx:        fa.CellKField[ta.wpfloat],             # Temporary field
    dqx:       fa.CellKField[ta.wpfloat],             # Temporary field
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LVC:       ta.wpfloat,
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
    qve_out:   fa.CellKField[ta.wpfloat],             # Specific humidity
    qce_out:   fa.CellKField[ta.wpfloat],             # Specific cloud water content
    te_out:    fa.CellKField[ta.wpfloat],             # Temperature
):
    _satadj_init( te, qve, qce, qre, qti, CI, CLW, CVD, CVV, LVC, out=(cvc, ue, Tx_hold, Tx) )
    _qsat_rho(Tx_hold, rho, TMELT, RV, out=qx_hold)

    # Newton-Raphson iteration
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)
    _dqsatdT_rho(qx, Tx, TMELT, out=dqx)
    _newton_raphson(qx, dqx, Tx, rho, qve, qce, cvc, ue, CVV, CLW, LVC, TMELT, RV, out=Tx)

    # final humidity calculation
    _qsat_rho(Tx, rho, TMELT, RV, out=qx)

    # final calculation of output variables
    _output_calculation( qve, qce, qx_hold, qx, Tx_hold, Tx, out=(te_out, qve_out, qce_out) )
