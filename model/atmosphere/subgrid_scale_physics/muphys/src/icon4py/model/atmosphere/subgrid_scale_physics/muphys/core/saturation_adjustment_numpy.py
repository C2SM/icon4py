# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


C1ES = ta.wpfloat(610.78)
C3LES = ta.wpfloat(17.269)
C4LES = ta.wpfloat(35.86)
C5LES = C3LES * (ThermodynamicConsts.tmelt - C4LES)


def _qsat_rho_numpy(t: data_alloc.NDArray, rho: data_alloc.NDArray) -> data_alloc.NDArray:
    return (C1ES * np.exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))) / (
        rho * ThermodynamicConsts.rv * t
    )


def _dqsatdT_rho_numpy(qs: data_alloc.NDArray, t: data_alloc.NDArray) -> data_alloc.NDArray:
    return qs * (C5LES / ((t - C4LES) * (t - C4LES)) - ta.wpfloat(1.0) / t)


def _newton_raphson_numpy(
    Tx: data_alloc.NDArray,
    rho: data_alloc.NDArray,
    qve: data_alloc.NDArray,
    qce: data_alloc.NDArray,
    cvc: data_alloc.NDArray,
    ue: data_alloc.NDArray,
) -> data_alloc.NDArray:
    qx = _qsat_rho_numpy(Tx, rho)
    dqx = _dqsatdT_rho_numpy(qx, Tx)
    qcx = qve + qce - qx
    cv = cvc + ThermodynamicConsts.cvv * qx + ThermodynamicConsts.clw * qcx
    ux = cv * Tx - qcx * GraupelConsts.lvc
    dux = cv + dqx * (GraupelConsts.lvc + (ThermodynamicConsts.cvv - ThermodynamicConsts.clw) * Tx)
    Tx = Tx - (ux - ue) / dux
    return Tx


def saturation_adjustment_numpy(
    te: fa.CellKField[ta.wpfloat], rho: fa.CellKField[ta.wpfloat], q_in: Q
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    # Convert input fields to numpy arrays
    te = data_alloc.as_numpy(te)
    rho = data_alloc.as_numpy(rho)
    qv = data_alloc.as_numpy(q_in.v)
    qc = data_alloc.as_numpy(q_in.c)
    qr = data_alloc.as_numpy(q_in.r)
    qs = data_alloc.as_numpy(q_in.s)
    qi = data_alloc.as_numpy(q_in.i)
    qg = data_alloc.as_numpy(q_in.g)

    qti = qs + qi + qg
    qt = qv + qc + qr + qti

    cvc = (
        ThermodynamicConsts.cvd * (1.0 - qt) + ThermodynamicConsts.clw * qr + GraupelConsts.ci * qti
    )
    cv = cvc + ThermodynamicConsts.cvv * qv + ThermodynamicConsts.clw * qc
    ue = cv * te - qc * GraupelConsts.lvc

    Tx_hold = ue / (cv + qc * (ThermodynamicConsts.cvv - ThermodynamicConsts.clw))
    qx_hold = _qsat_rho_numpy(Tx_hold, rho)

    Tx = te.copy()

    # Newton-Raphson iteration: 6 times the same operations
    for _ in range(6):
        Tx = _newton_raphson_numpy(Tx, rho, qv, qc, cvc, ue)

    qx = _qsat_rho_numpy(Tx, rho)

    mask = qv + qc <= qx_hold
    te_out = np.where(mask, Tx_hold, Tx)
    qce_out = np.where(mask, 0.0, np.maximum(qv + qc - qx, 0.0))
    qve_out = np.where(mask, qv + qc, qx)

    return te_out, qve_out, qce_out
