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

import warnings

import numpy as np
from functional.iterator.embedded import np_as_located_field
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import arrays

from icon4py.atm_phy_schemes.mo_convect_tables import (
    c1es,
    c3les,
    c4les,
    c5les,
)

from icon4py.atm_phy_schemes.mo_satad import satad
from icon4py.common.dimension import CellDim, KDim
from icon4py.shared.mo_physical_constants import (
    alv,
    clw,
    cvd,
    rv,
    tmelt,
)
from icon4py.testutils.simple_mesh import SimpleMesh


def random_field_strategy(mesh, *dims) -> st.SearchStrategy[float]:
    """Return a hypothesis strategy of a random field."""
    return arrays(
        dtype=np.float64,
        shape=tuple(map(lambda x: mesh.size[x], dims)),
        elements=st.floats(
            min_value=0.0,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=True,
        ),
    ).map(np_as_located_field(*dims))


cp_v = 1850.0
ci = 2108.0

tol = 1e-3
maxiter = 10
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


def satad_numpy(qv, qc, t, rho):

    lwdocvd = latent_heat_vaporization(t) / cvd

    for cell, k in np.ndindex(np.shape(qv)):
        if qv[cell, k] + qc[cell, k] <= qsat_rho(t, rho)[cell, k]:
            qv[cell, k] = qv[cell, k] + qc[cell, k]
            t[cell, k] = t[cell, k] - lwdocvd[cell, k] * qc[cell, k]
            qc[cell, k] = 0.0
        else:
            twork = t[cell, k]
            tworkold = twork + 10.0

            for _ in 1, maxiter:
                if abs(twork - tworkold) > tol:
                    tworkold = twork
                    qwd = qsat_rho(t, rho)[cell, k]
                    dqwd = dqsatdT_rho(twork, qwd)
                    fT = twork - t[cell, k] + lwdocvd[cell, k] * (qwd - qv[cell, k])
                    dfT = 1.0 + lwdocvd[cell, k] * dqwd
                    twork = twork - fT / dfT

            t[cell, k] = twork

            qwa = qsat_rho(t[cell, k], rho[cell, k])
            qc[cell, k] = max(qc[cell, k] + qv[cell, k] - qwa, zqwmin)
            qv[cell, k] = qwa

            del twork
    return t, qv, qc


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim),
    random_field_strategy(SimpleMesh(), CellDim, KDim),
    random_field_strategy(SimpleMesh(), CellDim, KDim),
    random_field_strategy(SimpleMesh(), CellDim, KDim),
)
@settings(
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    deadline=None,
    max_examples=10,
)
def test_mo_satad(qv, qc, t, rho):

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            t_ref, qv_ref, qc_ref = satad_numpy(
                np.asarray(qv),
                np.asarray(qc),
                np.asarray(t),
                np.asarray(rho),
            )
        except RuntimeWarning:
            assume(False)

        # Exploit hypothesis tool to guess needed co-variability of inputs.
        try:
            tendency = np.asarray(qv) - qv_ref
            target(np.std(tendency), label="Stdev. tendency")
        except Exception:
            assume(False)

    satad(
        qv,
        qc,
        t,
        rho,
        offset_provider={},
    )

    assert np.allclose(t, t_ref)
    assert np.allclose(qv, qv_ref)
    assert np.allclose(qc, qc_ref)
