# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    saturation_adjustment,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid

C1ES = wpfloat(610.78)
C3LES = wpfloat(17.269)
C4LES = wpfloat(35.86)
C5LES = C3LES * (ThermodynamicConsts.tmelt - C4LES)


def _qsat_rho_numpy(t: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return (C1ES * np.exp(C3LES * (t - ThermodynamicConsts.tmelt) / (t - C4LES))) / (
        rho * ThermodynamicConsts.rv * t
    )


def _dqsatdT_rho_numpy(qs: np.ndarray, t: np.ndarray) -> np.ndarray:
    return qs * (C5LES / ((t - C4LES) * (t - C4LES)) - wpfloat(1.0) / t)


def _newton_raphson_numpy(
    Tx: np.ndarray,
    rho: np.ndarray,
    qve: np.ndarray,
    qce: np.ndarray,
    cvc: np.ndarray,
    ue: np.ndarray,
) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class TestSaturationAdjustment(StencilTest):
    PROGRAM = saturation_adjustment
    OUTPUTS = ("te_out", "qve_out", "qce_out")

    @staticmethod
    def reference(
        grid: base_grid.Grid,
        te: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(
            te_out=np.full(te.shape, 273.91226488486984),
            qve_out=np.full(te.shape, 4.4903852062454690e-003),
            qce_out=np.full(te.shape, 9.5724552280369163e-007),
        )

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        return dict(
            te=data_alloc.constant_field(
                grid, 273.90911754406039, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            q_in=Q(
                v=data_alloc.constant_field(
                    grid, 4.4913424511676030e-003, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                c=data_alloc.constant_field(
                    grid, 6.0066941654987605e-013, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                r=data_alloc.constant_field(
                    grid, 2.5939378002267028e-004, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                s=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                i=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                g=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
            ),
            rho=data_alloc.constant_field(
                grid, 1.1371657035251757, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            te_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qve_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qce_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
