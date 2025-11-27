# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import g_ct, t_d
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta

class Q(NamedTuple):
    v: fa.CellKField[ta.wpfloat]  # Specific humidity                                                                                                          
    c: fa.CellKField[ta.wpfloat]  # Specific cloud water content                                                                                               
    r: fa.CellKField[ta.wpfloat]  # Specific rain water                                                                                                        
    s: fa.CellKField[ta.wpfloat]  # Specific snow water                                                                                                        
    i: fa.CellKField[ta.wpfloat]  # Specific ice water content                                                                                                 
    g: fa.CellKField[ta.wpfloat]  # Specific graupel water content                                                                                             

@gtx.field_operator
def _saturation_adjustment(
    te: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    q_in: Q
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """
    Compute the saturation adjustment which revises internal energy and water contents

    Args:
        Tx:                    Temperature
        rho:                   Density containing dry air and water constituents
        Q:                     Class with humidity, cloud, rain, snow, ice and graupel water

    Result:                    Tuple containing
                               - Revised temperature
                               - Revised specific cloud water content
                               - Revised specific vapor content
    """
    qti = q_in.s + qie + qge
    qt = qve + qce + qre + qti
    cvc = t_d.cvd * (1.0 - qt) + t_d.clw * qre + g_ct.ci * qti
    cv = cvc + t_d.cvv * qve + t_d.clw * qce
    ue = cv * te - qce * g_ct.lvc
    Tx_hold = ue / (cv + qce * (t_d.cvv - t_d.clw))
    qx_hold = _qsat_rho(Tx_hold, rho)

    Tx = te
    # Newton-Raphson iteration: 6 times the same operations
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)
    Tx = _newton_raphson(Tx, rho, qve, qce, cvc, ue)

    # At this point we hope Tx has converged
    qx = _qsat_rho(Tx, rho)

    # Is it possible to unify the where for all three outputs??
    mask = qve + qce <= qx_hold
    te = where(mask, Tx_hold, Tx)
    qce = where(mask, 0.0, maximum(qve + qce - qx, 0.0))
    qve = where(mask, qve + qce, qx)

    return te, qve, qce


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def saturation_adjustment(
    te: fa.CellKField[ta.wpfloat],       # Temperature
    rho: fa.CellKField[ta.wpfloat],      # Density containing dry air and water constituents
    q_in: Q,                             # Class with humidity, cloud, rain, snow, ice and graupel water
    te_out: fa.CellKField[ta.wpfloat],   # Temperature
    qve_out: fa.CellKField[ta.wpfloat],  # Specific humidity
    qce_out: fa.CellKField[ta.wpfloat],  # Specific cloud water content
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _saturation_adjustment(
        te,
        rho,
        q_in,
        out=(te_out, qve_out, qce_out),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
