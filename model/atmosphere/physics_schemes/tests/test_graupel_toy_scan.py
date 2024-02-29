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
from gt4py.next.ffront.decorator import program, scan_operator
from gt4py.next.ffront.fbuiltins import Field
from hypothesis import given, settings

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field_strategy, zero_field


"""Do Implicit sedimentaion

TODO David:
- Organize IO

"""

MESH = SimpleGrid()


def graupel_toy_numpy(qc, qr):
    """Match this routine."""
    sedimentation = 0.0

    a = 0.1
    b = 0.2
    precipitation = np.zeros(qc.shape[0])
    for cell, k in np.ndindex(qc.shape):
        # Autoconversion: Cloud Drops -> Rain Drops
        qc[cell, k] -= qc[cell, k] * a
        qr[cell, k] += qc[cell, k] * a

        # Add sedimentation from gridpoint above
        qr[cell, k] += sedimentation

        # Compute new sedimentation rate (made up formula).
        sedimentation = b * qr[cell, k - 1] ** 3 if qr[cell, k] <= 0.1 else 0.0

        # Precipitation is qr arriving at the ground
        if k != qc.shape[1]:
            qr[cell, k] -= sedimentation
        else:
            precipitation[cell] = sedimentation

    return precipitation


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, 0.0))
def _graupel_toy_scan(
    carry: tuple[float, float, float],
    qc_in: float,
    qr_in: float,
) -> tuple[float, float, float]:

    a = 0.1
    b = 0.2

    # unpack carry
    sedimentation_kMinus1, qr_kMinus1, _ = carry

    # Autoconversion: Cloud Drops -> Rain Drops
    qc = qc_in - qc_in * a
    qr = qc_in + qc_in * a

    # Add sedimentation from above level
    qr = qr + sedimentation_kMinus1

    # Compute new sedimentation rate (made up formula).
    sedimentation = b * qr_kMinus1**3 if qr <= 0.1 else 0.0

    # if k is not np.shape(qc)[1]: #TODO: Need if on index field
    qr = qr_in - sedimentation

    return sedimentation, qr, qc


@program
def graupel_toy_scan(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
    sedimentation: Field[[CellDim, KDim], float],
):
    # Writing to several output fields currently breaks due to gt4py bugs
    _graupel_toy_scan(qc, qr, out=(sedimentation, qr, qc))


@given(
    random_field_strategy(MESH, CellDim, KDim, min_value=1e-7, max_value=1),
    random_field_strategy(MESH, CellDim, KDim, min_value=1e-7, max_value=1),
)
@settings(deadline=None, max_examples=10)
def test_graupel_toy_scan(qc, qr):

    mesh = MESH

    sedimentation = zero_field(mesh, CellDim, KDim)
    qc_numpy = np.asarray(qc)
    qr_numpy = np.asarray(qr)

    graupel_toy_scan(qc, qr, sedimentation, offset_provider={})
    precipitation_numpy = graupel_toy_numpy(qc_numpy, qr_numpy)

    assert np.allclose(np.asarray(qc), qc_numpy)
    assert np.allclose(np.asarray(qr), qr_numpy)
    assert np.allclose(np.asarray(sedimentation)[:, 0], precipitation_numpy)
