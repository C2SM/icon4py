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


from functional.ffront.decorator import program, scan_operator
from functional.ffront.fbuiltins import Field
from hypothesis import given, settings

from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field_strategy


"""Implicit sedimentaion"""

"""

TODO GT4Py team:
- ternary in scan_operator (Till, fast track)
- local IF statement in scan_operator --> big ticket item (Till)
- think about optimized returning CellDim field (hotfix Till)
- Cant passs scalars to scan()


TODO DL:
- Make compile with changed if
- Recode in Python -> test scan operator
- send code snippet to Till
"""


# DL: Init initiates the state?
@scan_operator(axis=KDim, forward=True, init=0.0)
def _graupel_scan(state: float, qc: float) -> float:

    a = 0.1
    return qc + state * a


@program
def graupel(
    qc_before: Field[[CellDim, KDim], float],
    qc: Field[[CellDim, KDim], float],
):
    _graupel_scan(qc_before, out=qc)


@given(
    random_field_strategy(SimpleMesh(), CellDim, KDim),
    random_field_strategy(SimpleMesh(), CellDim, KDim),
)
@settings(deadline=None, max_examples=10)
def test_graupel(qc_before, qc):

    graupel(qc_before, qc, offset_provider={"Koff": KDim})


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


# @settings(max_examples=100)
# @given(
#     random_field_strategy(SimpleMesh(), CellDim, KDim),
#     random_field_strategy(SimpleMesh(), CellDim, KDim),
#     random_field_strategy(SimpleMesh(), CellDim, KDim),
#     random_field_strategy(SimpleMesh(), KDim),
#     st.floats(),
#     st.floats(),
# )
# def test_graupel_stub(qc, qr, precipitation, sedimentation, a, b):
#     graupel_stub(qc, qr, precipitation, a, b, offset_provider={"Koff": KDim})
