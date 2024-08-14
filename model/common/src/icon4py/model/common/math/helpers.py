# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import Field, field_operator

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2V, KDim, Koff, VertexDim
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
        half_level_field: Field[Dims[CellDim, KDim], wpfloat]

    Returns: Field[Dims[CellDim, KDim], wpfloat] full level field

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
        half_level_field: Field[Dims[CellDim, KDim], wpfloat]

    Returns: Field[Dims[CellDim, KDim], wpfloat] full level field

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
        half_level_field: Field[Dims[CellDim, KDim], wpfloat]

    Returns: Field[Dims[CellDim, KDim], wpfloat] full level field

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
    psi_v: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: fa.EdgeField[float],
    tangent_orientation: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e
