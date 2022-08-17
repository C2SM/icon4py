# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np
from hypothesis import given, settings

from icon4py.atm_phy_schemes.mo_convect_tables import c1es, c3les, c4les, c5les
from icon4py.atm_phy_schemes.mo_satad import _newtonian_iteration_t, satad
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import alv, clw, cvd, rv, tmelt
from icon4py.testutils.simple_mesh import (
    SimpleMesh,
    maximizeTendency,
    random_field_strategy,
)


cp_v = 1850.0
ci = 2108.0

tol = 1e-3
maxiter = 1  # DL: Needs to be 10, but GT4Py is currently too slow
zqwmin = 1e-20


def latent_heat_vaporization(t):
    return alv + (cp_v - clw) * (t - tmelt) - rv * t


def sat_pres_water(t):
    return c1es * np.exp(c3les * (t - tmelt) / (t - c4les))


def qsat_rho(t, rho):
    return sat_pres_water(t) / (rho * rv * t)


def dqsatdT_rho(t, zqsat):
    beta = c5les / (t - c4les) ** 2 - 1.0 / t
    return beta * zqsat


def newtonian_iteration_t(t, qv, rho):
    lwdocvd = latent_heat_vaporization(t) / cvd

    tWork = t.copy()
    tWorkold = tWork.copy() + 10.0

    for _ in range(maxiter):
        if abs(tWork - tWorkold) > tol:
            tWorkold = tWork
            qwd = qsat_rho(t, rho)
            dqwd = dqsatdT_rho(tWork, qwd)
            fT = tWork - t + lwdocvd * (qwd - qv)
            dfT = 1.0 + lwdocvd * dqwd
            tWork = tWork - fT / dfT

    return tWork


def satad_numpy(qv, qc, t, rho):

    lwdocvd = latent_heat_vaporization(t) / cvd

    for cell, k in np.ndindex(np.shape(qv)):
        totallySubsaturated = qv[cell, k] + qc[cell, k] <= qsat_rho(t, rho)[cell, k]

        if totallySubsaturated:
            t[cell, k] = t[cell, k] - lwdocvd[cell, k] * qc[cell, k]
            qv[cell, k] = qv[cell, k] + qc[cell, k]
            qc[cell, k] = 0.0
        else:
            t[cell, k] = newtonian_iteration_t(t[cell, k], qv[cell, k], rho[cell, k])

            qwa = qsat_rho(t[cell, k], rho[cell, k])
            qc[cell, k] = max(qc[cell, k] + qv[cell, k] - qwa, zqwmin)
            qv[cell, k] = qwa

    return t, qv, qc


# TODO: Understand magic number 1e-8. Single precision-related?
@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=200, max_value=350),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
)
@settings(deadline=None, max_examples=10)
def test_newtonian_iteration(t, qv, rho):
    tRef = np.zeros_like(np.asarray(t))

    for cell, k in np.ndindex(np.shape(t)):
        tRef[cell, k] = newtonian_iteration_t(
            np.asarray(t).copy()[cell, k],
            np.asarray(qv)[cell, k],
            np.asarray(rho)[cell, k],
        )
    _newtonian_iteration_t(t, qv, rho, out=t, offset_provider={})

    maximizeTendency(t, tRef, "t")
    assert np.allclose(np.asarray(t), tRef)


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=200, max_value=350),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
)
@settings(deadline=None, max_examples=10)
def test_mo_satad(qv, qc, t, rho):

    tRef, qvRef, qcRef = satad_numpy(
        np.asarray(qv).copy(),
        np.asarray(qc).copy(),
        np.asarray(t).copy(),
        np.asarray(rho).copy(),
    )

    maximizeTendency(t, tRef, "t")
    maximizeTendency(qv, qvRef, "qv")
    maximizeTendency(qc, qcRef, "qc")

    satad(qv, qc, t, rho, offset_provider={})

    assert np.allclose(np.asarray(t), tRef)
    assert np.allclose(np.asarray(qv), qvRef)
    assert np.allclose(np.asarray(qc), qcRef)
