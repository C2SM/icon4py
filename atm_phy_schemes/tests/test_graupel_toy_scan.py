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
from functional.ffront.decorator import program, scan_operator
from hypothesis import given, settings

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


def graupel_toy_numpy(qc, qr, precipitaion, a=0.1, b=0.2):
    """Current goal is ot match this routine."""
    sedimentation = 0.0

    for cell, k in np.ndindex(np.shape(qc)):

        # Autoconversion: Cloud Drops -> Rain Drops
        qc[cell, k] -= qc[cell, k] * a
        qr[cell, k] += qc[cell, k] * a

        # Add sedimentation from gridpoint above
        qr[cell, k] += sedimentation

        # Compute new sedimentation rate (made up formula).
        sedimentation = b * qr[cell, k - 1] ** 3 if qr[cell, k] <= 0.1 else 0.0

        # Precipitation is qr arriving at the ground
        if k is not np.shape(qc)[1]:
            qr[cell, k] -= sedimentation
        else:
            precipitaion[cell, k] = qr[cell, k]

    return precipitaion


@scan_operator(axis=KDim, forward=True, init=0.0)
def _graupel_scan_operator(state: float, qc: float, qr: float) -> float:

    a = 0.1
    return state + a


# @given(
#     random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-7, max_value=1),
#     random_field_strategy(SimpleMesh(), CellDim, KDim, min_value=1e-7, max_value=1),
# )
# @settings(deadline=None, max_examples=10)
# def test_graupel(qc, qr):

#     mesh = SimpleMesh()
#     qc_out = zero_field(mesh, CellDim, KDim)
#     qr_out = zero_field(mesh, CellDim, KDim)

#     graupel(qc, qr, qc_out, qr_out, offset_provider={"Koff": KDim})

#     precipitation_out_numpy = np.asarray(zero_field(mesh, CellDim))
#     qc_out_numpy = np.asarray(qc_out).copy()
#     qr_out_numpy = np.asarray(qr_out).copy()

#     qc_out_numpy, qr_out_numpy, precipitation_out_numpy = graupel_numpy(
#         np.asarray(qc), np.asarray(qr)
#     )  # Changes qc, qr
#     maximizeTendency(precipitation_out_numpy, 0.0, "precipitation")


# @scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, 0.0))
# def _graupel_stub(
#     state: tuple[float, float, float],
#     # Prognostic 3D Fields -> These are INOUT!
#     qc: float,  # Field
#     qr: float,  # Fieldmade up if
#     precipitation: float,
#     # Scalars
#     a: float,
#     b: float,
# ) -> tuple[float, float, float]:

#     # Unwrap state
#     qc_up, qr_up, precipitation_up = state

#     # # Autoconversion: Cloud Drops -> Rain Drops
#     # qc = qc - qc * a
#     # qr = qr + qc * a

#     # # Rain sedimentation (contribution from gridbox above)
#     # qr = qr + precipitation_up

#     # # Compute new sedimentation rate. Depends on qc value below
#     # # in F90 there is an (if k /= ke) here
#     # # We need to replace this with two compute regions.
#     # # The second region doesnt have the feature blow
#     # precipitation = 1e-6  # IF(qr_up <= 0.1, 0.0, b * qr_up ** 3)

#     # qr = qr - precipitation

#     return (qc, qr, precipitation)


# @program
# def graupel_stub(
#     qc: Field[CellDim, KDim, float],
#     qr: Field[CellDim, KDim, float],
#     rain_sedimentation: Field[CellDim, float],
# ):
#     a = 1
#     b = 2
#     _graupel_stub(
#         qc,
#         qr,
#         a,
#         b,
#         out=(
#             qc,
#             qr,
#             rain_sedimentation,
#         ),
#     )

#     # precipitation = rain_sedimentation[last]



