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
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, where
from hypothesis import given, settings
from test_graupel_toy_scan import graupel_toy_numpy

from icon4py.common.dimension import CellDim, KDim, Koff
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field_strategy, zero_field


"""Implicit sedimentaion"""

"""

TODO GT4Py team:
- ternary in scan_operator (Niki, fast track)
- local IF statement in scan_operator --> big ticket item (Till)
- think about optimized returning CellDim field (hotfix Till)
- Pass scalars to scan()
- Need index field for field_operator version


To discuss:
- Fucntional model in, out, overwrite
- Multiple returns from scan_operator?
- Return 3D field?


TODO DL:
- Make compile
- Recode in Python -> test scan operator
- send code snippet to Till

Questions:
1. How and when to switch qc in and qc_out? -> Hannes nees to explain me functional programming


Questions for field_operator version
1.  Cant return both 2D and 3D tuples in field operator. "Incompatible fields in tuple: all fields must have the same dimensions." Is this intentional?
"""


@field_operator
def _graupel_toy_field_operator(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
    precipitation: Field[[CellDim, KDim], float],  # DL: TODO: This should be 2D
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],  # DL: TODO: This should be 2D
]:

    a = 0.1
    b = 0.2

    # Autoconversion: Cloud Drops -> Rain Drops
    qc = qc - qc * a
    qr = qr + qc * a

    # Add sedimentation from gridpoint above
    qr = qr + precipitation(Koff[-1])

    # Compute new sedimentation rate (made up formula).
    precipitation = where(qr <= 0.1, b * qr(Koff[-1]) ** 3, 0.0)

    # # Precipitation is qr arriving at the ground TODO: Need index field
    # if k is not np.shape(qc)[1]:
    #     qr[cell, k] -= precipitation

    return (qc, qr, precipitation)


@program()
def graupel_toy_field_operator(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
    precipitation: Field[[CellDim, KDim], float],
):
    _graupel_toy_field_operator(qc, qr, precipitation, out=(qc, qr, precipitation))


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-7, max_value=1),
    random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-7, max_value=1),
)
@settings(deadline=None, max_examples=10)
def test_graupel_field_operator(qc, qr):

    mesh = SimpleMesh()
    precipitation = zero_field(mesh, CellDim, KDim)

    qc_numpy = np.asarray(qc).copy()
    qr_numpy = np.asarray(qc).copy()
    precipitation_numpy = np.asarray(precipitation).copy()[:, -1]

    # qc_numpy and qr_numpy are modified in place
    precipitation_numpy = graupel_toy_numpy(qc_numpy, qr_numpy, precipitation_numpy)

    graupel_toy_field_operator(qc, qr, precipitation, offset_provider={"Koff": KDim})

    assert np.allclose(np.asarray(qc), qc_numpy)
    assert np.allclose(np.asarray(qr), qr_numpy)
    assert np.allclose(np.asarray(precipitation)[:, -1], precipitation_numpy)
