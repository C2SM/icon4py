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

from functional.ffront.decorator import program, field_operator, scan_operator
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field_strategy, zero_field


"""Do Implicit sedimentaion


TODO GT4Py team:
- ternary in scan_operator (Niki, fast track)
- local IF statement in scan_operator --> big ticket item (Till)
- think about optimized returning CellDim field (hotfix Till)
- Pass scalars to scan()
- Need index field for field_operator version


To discuss:
- Returns from scan. Carry, carry + return, 2D Fields in Return

TODO DL:
- Make compile
- Recode in Python -> test scan operator
- send code snippet to Till

Questions:
1. How and when to switch qc in and qc_out? -> Hannes nees to explain me functional programming


Questions for field_operator version
1.  Cant return both 2D and 3D tuples in field operator. "Incompatible fields in tuple: all fields must have the same dimensions." Is this intentional?
"""

MESH = SimpleMesh()


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
        sedimentation = b * qr[cell, k - 1] ** 3  # if qr[cell, k] <= 0.1 else 0.0

        # Precipitation is qr arriving at the ground
        if k != qc.shape[1]:
            qr[cell, k] -= sedimentation
        else:
            precipitation[cell] = sedimentation


    return precipitation


@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, 0.0))
def _graupel_toy_scan(
    carry: tuple[float, float, float], qc_in: float, qr_in: float
) -> tuple[float, float, float]:

    a = 0.1
    b = 0.2

    # unpack carry
    # TODO Discussion: Split current carry return into carry + return ? Here we dont touch qc_kMinus1
    # Also: Discuss syntax
    sedimentation_kMinus1, qc_kMinus1, qr_kMinus1 = carry

    # Autoconversion: Cloud Drops -> Rain Drops
    qc = qc_in - qc_in * a
    qr = qc_in + qc_in * a

    # Add sedimentation from gridpoint above
    qr = qr + sedimentation_kMinus1

    # Compute new sedimentation rate (made up formula).
    sedimentation = b * qr_kMinus1**3  # if qr[cell, k] <= 0.1 else 0.0

    # if k is not np.shape(qc)[1]: #TODO: Need if on index field
    qr = qr_in - sedimentation

    return (sedimentation, qc, qr)


@field_operator
def _graupel_toy_fo(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
) -> Field[CellDim, KDim]:
    sedimentation, qc, qr = _graupel_toy_scan(qc, qr)
    return sedimentation


@program
def graupel_toy_scan(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
    sedimentation: Field[[CellDim, KDim], float],
):
    # Writing to several output fields currently breaks due to gt4py bugs
    _graupel_toy_fo(qc, qr, out=sedimentation)


@given(
    random_field_strategy(MESH, CellDim, KDim, min_value=1e-7, max_value=1),
    random_field_strategy(MESH, CellDim, KDim, min_value=1e-7, max_value=1),
)
@settings(deadline=None, max_examples=10)
def test_graupel_scan(qc, qr):
    mesh = MESH
    sedimentation = zero_field(mesh, CellDim, KDim)

    qc_numpy = np.asarray(qc).copy()
    qr_numpy = np.asarray(qc).copy()

    graupel_toy_scan(qc, qr, sedimentation, offset_provider={})

    # qc_numpy and qr_numpy are modified in place
    precipitation_numpy = graupel_toy_numpy(qc_numpy, qr_numpy)

    # Uncomment the following assert once gt4py fixes the bug
    # when returning multiple fields
    # assert np.allclose(np.asarray(qc), qc_numpy)
    # assert np.allclose(np.asarray(qr), qr_numpy)
    # TODO: Understand why the k order is the opposite of the expected order
    assert np.allclose(np.asarray(sedimentation)[:, 0], precipitation_numpy)
