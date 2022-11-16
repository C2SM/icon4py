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


import contextlib
import io
import trace

import numpy as np
from hypothesis import given, settings, target

from icon4py.atm_phy_schemes.mo_convect_tables import conv_table
from icon4py.atm_phy_schemes.mo_satad import _newtonian_iteration_temp, satad
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import phy_const
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import maximizeTendency, random_field_strategy


cp_v = 1850.0
ci = 2108.0

tol = 1e-3
maxiter = 2  # DL: Needs to be 10, but GT4Py is currently too slow
zqwmin = 1e-20


def latent_heat_vaporization(t):
    """Return latent heat of vaporization given a temperature."""
    return (
        phy_const.alv
        + (cp_v - phy_const.clw) * (t - phy_const.tmelt)
        - phy_const.rv * t
    )


def sat_pres_water(t):
    """Saturation pressure of water."""
    return conv_table.c1es * np.exp(
        conv_table.c3les * (t - phy_const.tmelt) / (t - conv_table.c4les)
    )


def qsat_rho(t, rho):
    return sat_pres_water(t) / (rho * phy_const.rv * t)


def dqsatdT_rho(t, zqsat):
    """Return derivative of qsat with respect to t."""
    beta = conv_table.c5les / (t - conv_table.c4les) ** 2 - 1.0 / t
    return beta * zqsat


def newtonian_iteration_temp(t, twork, tworkold, qv, rho):
    """Obtain temperature at saturation using Newtonian iteration."""
    lwdocvd = latent_heat_vaporization(t) / phy_const.cvd

    for _ in range(maxiter):
        if abs(twork - tworkold) > tol:
            tworkold = twork
            qwd = qsat_rho(twork, rho)
            dqwd = dqsatdT_rho(twork, qwd)
            fT = twork - t + lwdocvd * (qwd - qv)
            dfT = 1.0 + lwdocvd * dqwd
            twork = twork - fT / dfT

    return twork


def satad_numpy(qv, qc, t, rho):
    """Numpy translation of satad_v_3D from Fortan ICON."""
    for cell, k in np.ndindex(np.shape(qv)):
        lwdocvd = latent_heat_vaporization(t[cell, k]) / phy_const.cvd

        Ttest = t[cell, k] - lwdocvd * qc[cell, k]

        if qv[cell, k] + qc[cell, k] <= qsat_rho(Ttest, rho[cell, k]):
            qv[cell, k] = qv[cell, k] + qc[cell, k]
            qc[cell, k] = 0.0
            t[cell, k] = Ttest
        else:
            t[cell, k] = newtonian_iteration_temp(
                t[cell, k], t[cell, k], t[cell, k] + 10.0, qv[cell, k], rho[cell, k]
            )

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
    """Test newtonian_iteration aginst a numpy implementaion."""
    tRef = np.zeros_like(np.asarray(t))

    # Numpy Implementation
    for cell, k in np.ndindex(np.shape(t)):

        tRef[cell, k] = newtonian_iteration_temp(
            np.asarray(t)[cell, k],
            np.asarray(t)[cell, k],
            np.asarray(t)[cell, k] + 10.0,
            np.asarray(qv)[cell, k],
            np.asarray(rho)[cell, k],
        )

    # Guide hypothesis tool to maximize tendency of t
    maximizeTendency(t, tRef, "t")

    # GT4Py Implementation
    _newtonian_iteration_temp(t, qv, rho, out=t, offset_provider={})

    assert np.allclose(np.asarray(t), tRef)


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=200, max_value=350),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-8, max_value=1.0),
)
@settings(deadline=None, max_examples=10)
def test_mo_satad(qv, qc, t, rho):
    """Test satad aginst a numpy implementaion."""
    # Numpy Implementation
    tracer = trace.Trace(trace=1, count=1)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        tRef, qvRef, qcRef = tracer.runfunc(
            satad_numpy,
            np.asarray(qv).copy(),
            np.asarray(qc).copy(),
            np.asarray(t).copy(),
            np.asarray(rho).copy(),
        )

    lines = len(
        [line_no for fname, line_no in tracer.counts.keys() if fname == __file__]
    )

    # Guide hypothesis to maximize the number of lines visited
    target(lines, label="lines")

    # Guide hypothesis tool to maximize tendencies
    maximizeTendency(t, tRef, "t")
    maximizeTendency(qv, qvRef, "qv")
    maximizeTendency(qc, qcRef, "qc")

    # GT4Py Implementation
    satad(qv, qc, t, rho, offset_provider={})

    # Check results using a tolerance test
    assert np.allclose(np.asarray(t), tRef)
    assert np.allclose(np.asarray(qv), qvRef)
    assert np.allclose(np.asarray(qc), qcRef)
