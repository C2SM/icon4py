# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py import next as gtx
from gt4py.next import Field, field_operator
from gt4py.next.ffront.fbuiltins import cos, sin, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2V, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def average_cell_kdim_level_up(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def average_edge_kdim_level_up(
    half_level_field: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over an edge field for storage
    in the corresponding full levels.
    Args:
        half_level_field: fa.EdgeKField[wpfloat]

    Returns: fa.EdgeKField[wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def difference_k_level_down(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels downwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field(Koff[-1]) - half_level_field


@field_operator
def difference_k_level_up(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field - half_level_field(Koff[1])


@field_operator
def grad_fd_norm(
    psi_c: fa.CellKField[float],
    inv_dual_edge_length: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    """
    Calculate the gradient value of adjacent interface levels.

    Computes the difference of two offseted values multiplied by a field of the offseted dimension
    Args:
        psi_c: fa.CellKField[float],
        inv_dual_edge_length: Field[Dims[EdgeDim], float],

    Returns: fa.EdgeKField[float]

    """
    grad_norm_psi_e = (psi_c(E2C[1]) - psi_c(E2C[0])) * inv_dual_edge_length
    return grad_norm_psi_e


@field_operator
def _grad_fd_tang(
    psi_v: Field[[dims.VertexDim, dims.KDim], float],
    inv_primal_edge_length: fa.EdgeField[float],
    tangent_orientation: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e


@gtx.field_operator
def spherical_to_cartesian_on_cells(
    lat: fa.CellField[ta.wpfloat], lon: fa.CellField[ta.wpfloat], r: ta.wpfloat
) -> tuple[fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat]]:
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return x, y, z


@gtx.field_operator
def spherical_to_cartesian_on_edges(
    lat: fa.EdgeField[ta.wpfloat], lon: fa.EdgeField[ta.wpfloat], r: ta.wpfloat
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return x, y, z


@gtx.field_operator
def spherical_to_cartesian_on_vertex(
    lat: fa.VertexField[ta.wpfloat], lon: fa.VertexField[ta.wpfloat], r: ta.wpfloat
) -> tuple[fa.VertexField[ta.wpfloat], fa.VertexField[ta.wpfloat], fa.VertexField[ta.wpfloat]]:
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return x, y, z


@gtx.field_operator
def dot_product(
    x1: fa.EdgeField[ta.wpfloat],
    x2: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    y2: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    z2: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def norm2(
    x: fa.EdgeField[ta.wpfloat], y: fa.EdgeField[ta.wpfloat], z: fa.EdgeField[ta.wpfloat]
) -> fa.EdgeField[ta.wpfloat]:
    return sqrt(dot_product(x, x, y, y, z, z))


@gtx.field_operator
def normalize_cartesian_vector(
    v_x: fa.EdgeField[ta.wpfloat], v_y: fa.EdgeField[ta.wpfloat], v_z: fa.EdgeField[ta.wpfloat]
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    norm = norm2(v_x, v_y, v_z)
    return v_x / norm, v_y / norm, v_z / norm


@gtx.field_operator
def invert(f: fa.EdgeField[ta.wpfloat]) -> fa.EdgeField[ta.wpfloat]:
    return where(f != 0.0, 1.0 / f, f)
