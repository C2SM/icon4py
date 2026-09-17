# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor operations on unstructured grid fields.

Contains 3-vectors and 3x3 tensors whose components are edge fields on full levels,
with the transpose, trace, sum, symmetric part and squared norm defined on them.

A component is addressed by its index: ``m[1][2]`` is ``M_12``. What the three
indices mean is the caller's convention; for the velocity gradient of the ICON
grid they are the edge-normal, edge-tangential and vertical directions.
"""

from gt4py import next as gtx

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


Vec3OnEdges = tuple[fa.EdgeKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat]]
"""Three edge fields that form a vector at every edge and full level."""

Mat3OnEdges = tuple[Vec3OnEdges, Vec3OnEdges, Vec3OnEdges]
"""Three vectors that form a 3x3 tensor at every edge and full level."""


@gtx.field_operator
def transpose_on_edges(m: Mat3OnEdges) -> Mat3OnEdges:
    """Transpose: ``(M^t)_ij = M_ji``."""
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


@gtx.field_operator
def add_on_edges(a: Mat3OnEdges, b: Mat3OnEdges) -> Mat3OnEdges:
    """Componentwise sum ``A + B``."""
    return (
        (a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]),
        (a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]),
        (a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]),
    )


@gtx.field_operator
def trace_on_edges(m: Mat3OnEdges) -> fa.EdgeKField[ta.wpfloat]:
    """Trace ``M_ii``."""
    return m[0][0] + m[1][1] + m[2][2]


@gtx.field_operator
def diagonal_on_edges(m: Mat3OnEdges) -> Vec3OnEdges:
    """The diagonal ``(M_00, M_11, M_22)``."""
    return (m[0][0], m[1][1], m[2][2])


@gtx.field_operator
def upper_offdiagonal_on_edges(m: Mat3OnEdges) -> Vec3OnEdges:
    """The entries above the diagonal, ``(M_01, M_02, M_12)``."""
    return (m[0][1], m[0][2], m[1][2])


@gtx.field_operator
def dot_on_edges(a: Vec3OnEdges, b: Vec3OnEdges) -> fa.EdgeKField[ta.wpfloat]:
    """Dot product ``a_i b_i``."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@gtx.field_operator
def twice_symmetric_part_on_edges(m: Mat3OnEdges) -> Mat3OnEdges:
    """``M + M^t``, twice the symmetric part: ``(M + M^t)_ij = M_ij + M_ji``."""
    return add_on_edges(m, transpose_on_edges(m))


@gtx.field_operator
def squared_norm_of_symmetric_on_edges(m: Mat3OnEdges) -> fa.EdgeKField[ta.wpfloat]:
    """
    Squared norm ``M_ij M_ij`` of a symmetric tensor.

    Only the diagonal and the entries above it are read, so a tensor that is not
    symmetric gives a wrong result rather than an error.
    """
    d = diagonal_on_edges(m)
    u = upper_offdiagonal_on_edges(m)
    return dot_on_edges(d, d) + wpfloat("2.0") * dot_on_edges(u, u)
