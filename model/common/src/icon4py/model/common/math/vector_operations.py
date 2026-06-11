# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Vector operations on unstructured grid fields.

Contains dot product, cross product, norm and normalization operations
for vectors defined on cell, edge and vertex fields.
"""

from gt4py import next as gtx
from gt4py.next import sqrt

from icon4py.model.common import field_type_aliases as fa


@gtx.field_operator
def dot_product_on_edges(
    x1: fa.EdgeField[gtx.float64],
    x2: fa.EdgeField[gtx.float64],
    y1: fa.EdgeField[gtx.float64],
    y2: fa.EdgeField[gtx.float64],
    z1: fa.EdgeField[gtx.float64],
    z2: fa.EdgeField[gtx.float64],
) -> fa.EdgeField[gtx.float64]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_cells(
    x1: fa.CellField[gtx.float64],
    x2: fa.CellField[gtx.float64],
    y1: fa.CellField[gtx.float64],
    y2: fa.CellField[gtx.float64],
    z1: fa.CellField[gtx.float64],
    z2: fa.CellField[gtx.float64],
) -> fa.CellField[gtx.float64]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_vertices(
    x1: fa.VertexField[gtx.float64],
    x2: fa.VertexField[gtx.float64],
    y1: fa.VertexField[gtx.float64],
    y2: fa.VertexField[gtx.float64],
    z1: fa.VertexField[gtx.float64],
    z2: fa.VertexField[gtx.float64],
) -> fa.VertexField[gtx.float64]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def cross_product_on_edges(
    x1: fa.EdgeField[gtx.float64],
    x2: fa.EdgeField[gtx.float64],
    y1: fa.EdgeField[gtx.float64],
    y2: fa.EdgeField[gtx.float64],
    z1: fa.EdgeField[gtx.float64],
    z2: fa.EdgeField[gtx.float64],
) -> tuple[fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64]]:
    """Compute cross product of cartesian vectors (x1, y1, z1) x (x2, y2, z2)"""
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    return x, y, z


@gtx.field_operator
def norm2_on_edges(
    x: fa.EdgeField[gtx.float64], y: fa.EdgeField[gtx.float64], z: fa.EdgeField[gtx.float64]
) -> fa.EdgeField[gtx.float64]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_edges(x, x, y, y, z, z))


@gtx.field_operator
def norm2_on_cells(
    x: fa.CellField[gtx.float64], y: fa.CellField[gtx.float64], z: fa.CellField[gtx.float64]
) -> fa.CellField[gtx.float64]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_cells(x, x, y, y, z, z))


@gtx.field_operator
def norm2_on_vertices(
    x: fa.VertexField[gtx.float64], y: fa.VertexField[gtx.float64], z: fa.VertexField[gtx.float64]
) -> fa.VertexField[gtx.float64]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_vertices(x, x, y, y, z, z))


@gtx.field_operator
def normalize_cartesian_vector_on_edges(
    v_x: fa.EdgeField[gtx.float64], v_y: fa.EdgeField[gtx.float64], v_z: fa.EdgeField[gtx.float64]
) -> tuple[fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64]]:
    """
    Normalize a cartesian vector.

    Args:
        v_x: x coordinate
        v_y: y coordinate
        v_z: z coordinate

    Returns:
        normalized vector

    """
    norm = norm2_on_edges(v_x, v_y, v_z)
    return v_x / norm, v_y / norm, v_z / norm
