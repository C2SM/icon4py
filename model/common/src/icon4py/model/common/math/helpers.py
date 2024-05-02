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

from gt4py.next import Field, field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import E2C, E2V, CellDim, EdgeDim, KDim, Koff, VertexDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def average_cell_kdim_level_up(
    half_level_field: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[CellDim, KDim], wpfloat]

    Returns: Field[[CellDim, KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def average_edge_kdim_level_up(
    half_level_field: Field[[EdgeDim, KDim], wpfloat],
) -> Field[[EdgeDim, KDim], wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over an edge field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[EdgeDim, KDim], wpfloat]

    Returns: Field[[EdgeDim, KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def difference_k_level_down(
    half_level_field: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels downwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[CellDim, KDim], wpfloat]

    Returns: Field[[CellDim, KDim], wpfloat] full level field

    """
    return half_level_field(Koff[-1]) - half_level_field


@field_operator
def difference_k_level_up(
    half_level_field: Field[[CellDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[CellDim, KDim], wpfloat]

    Returns: Field[[CellDim, KDim], wpfloat] full level field

    """
    return half_level_field - half_level_field(Koff[1])


@field_operator
def grad_fd_norm(
    psi_c: Field[[CellDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    """
    Calculate the gradient value of adjacent interface levels.

    Computes the difference of two offseted values multiplied by a field of the offseted dimension
    Args:
        psi_c: Field[[CellDim, KDim], float],
        inv_dual_edge_length: Field[[EdgeDim], float],

    Returns: Field[[EdgeDim, KDim], float]

    """
    grad_norm_psi_e = (psi_c(E2C[1]) - psi_c(E2C[0])) * inv_dual_edge_length
    return grad_norm_psi_e


@field_operator
def _grad_fd_tang(
    psi_v: Field[[VertexDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
) -> Field[[EdgeDim, KDim], float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e


@field_operator
def _compute_inverse_edge_kdim(
    edge_k_field: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    return 1.0 / edge_k_field


@program
def compute_inverse_edge_kdim(
    edge_k_field: Field[[EdgeDim, KDim], vpfloat],
    inv_edge_k_field: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_inverse_edge_kdim(
        edge_k_field,
        out=inv_edge_k_field,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
